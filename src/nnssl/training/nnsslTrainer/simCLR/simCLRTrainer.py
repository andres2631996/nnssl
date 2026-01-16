from copy import deepcopy
from typing import Union, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from einops import rearrange

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from torch import autocast
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.voco_architecture import VoCoArchitecture
from nnssl.training.loss.contrastive_loss import NTXentLoss
from nnssl.utilities.helpers import dummy_context

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from nnssl.ssl_data.dataloading.simclr_transform import SimCLRTransform
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from nnssl.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from nnssl.utilities.helpers import empty_cache


class SimCLRTrainer(AbstractBaseTrainer):
    """
    TODO:
    - implement data aug path for simclr [x]
        - check which standard transforms to keep [x] - went with default nnUNet transforms fow now
    - re-use VoCoArchitecture (seems like no change necessary here, double-check) [x]
    - implement train/val steps (loss returns loss, accuracy) -> maybe track acc. similar to pseudo dice in nnUNet [x] - not tracking yet
    - re-implement similar to VoCoTransform (need more sub-crops, and random crops in general) [x]
    - maybe force partial overlaps between crops [x]
    - clean up, test runs [x]

    Memory consumption & batch/s on 4090:
    - batch_size 4, num_crops_per_image 3, crop_size (64, 64, 64): 9.55 GB & 5.3 batches/s
    - batch_size 8, num_crops_per_image 3, crop_size (64, 64, 64): 15.4 GB & 2.9 batches/s
    - batch_size 16, num_crops_per_image 3, crop_size (64, 64, 64): 23.5 GB & 1.08 batches/s
    - batch_size 32, num_crops_per_image 2, crop_size (64, 64, 64): >24.5 GB (OoM)
    - batch_size 32, num_crops_per_image 1, crop_size (64, 64, 64): 19.3 GB & 2.2 batches/s
    """

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple = (192, 192, 64),
        crop_size: tuple = (64, 64, 64),
        num_crops_per_image: int = 2,
        min_crop_overlap: float = 0.5,
    ):
        plan.configurations[configuration_name].patch_size = patch_size
        self.crop_size = crop_size

        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_crops_per_image = num_crops_per_image
        self.min_crop_overlap = min_crop_overlap

    def build_loss(self) -> nn.Module:
        """Implements the standard contrastive loss."""
        return NTXentLoss(
            batch_size=self.batch_size * self.num_crops_per_image,
            temperature=0.5,
            similarity_function="cosine",
            device=self.device,
        )

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
    ) -> AbstractTransform:
        tr_transforms = []

        if do_dummy_2d_data_aug:
            raise NotImplementedError(
                "We don't do dummy 2d aug here anymore. Data should be isotropic!"
            )

        # --------------------------- SimCLR Transformation --------------------------- #
        # All train augmentations are moved to the SimCLR Transform class.

        tr_transforms.append(
            SimCLRTransform(
                crop_size=self.crop_size,
                aug="train",
                crop_count_per_image=self.num_crops_per_image,
                min_overlap_ratio=self.min_crop_overlap,
                data_key="data",
            )
        )
        # From here on out we are working with reference and overlapping crops!

        tr_transforms.append(NumpyToTensor(["all_crops"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        val_transforms = []

        # --------------------------- VoCo Transformation --------------------------- #
        val_transforms.append(
            SimCLRTransform(
                crop_size=self.crop_size,
                aug="none",
                crop_count_per_image=self.num_crops_per_image,
                min_overlap_ratio=self.min_crop_overlap,
                data_key="data",
            )
        )

        val_transforms.append(NumpyToTensor(["all_crops"], "float"))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        # We don't do non-90 degree rotations for the VoCo Trainer.
        dl_tr, dl_val = self.get_plain_dataloaders(patch_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val

    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> nn.Module:
        encoder = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            encoder_only=True,
        )
        # Turns out VoCoArchitecture can be used for SimCLR purpose here.
        architecture = VoCoArchitecture(encoder, encoder.output_channels)

        plan = deepcopy(self.plan)
        plan.configurations[self.configuration_name].patch_size = self.crop_size

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=(
                "encoder.stem.convs.0.conv",
                "encoder.stem.convs.0.all_modules.0",
            ),
        )
        return architecture, adapt_plan

    def train_step(self, batch: Tuple[dict, dict]) -> dict:

        all_crops = batch["all_crops"]
        NREF = batch["reference_crop_index"]

        all_crops = all_crops.to(self.device, non_blocking=True)

        if torch.isnan(all_crops).any():
            print("NaN values found in input data!")
        if torch.isinf(all_crops).any():
            print("Infinity values found in input data!")

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            all_crop_embeddings = self.network(all_crops)
            if torch.isnan(all_crop_embeddings).any():
                print("NaN values found in embeddings!")

            # This rearrange below isn't necessary, but would come in handy when doing more involved contrastive tasks.
            # z_i_embeddings = rearrange(
            #     all_crop_embeddings[:NREF], "(b NREF) c -> b NREF c", b=self.batch_size
            # )
            # z_j_embeddings = rearrange(
            #     all_crop_embeddings[NREF:], "(b NREF) c -> b NREF c", b=self.batch_size
            # )

            # Normalize prior to contrastive loss
            z_i_embeddings = nn.functional.normalize(all_crop_embeddings[:NREF], dim=1)
            z_j_embeddings = nn.functional.normalize(all_crop_embeddings[NREF:], dim=1)

            # del data
            l, acc = self.loss(z_i_embeddings, z_j_embeddings)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # for name, param in self.network.named_parameters():
            #     if param.grad is not None:
            #         if param.grad.norm() > 1:
            #             print(f"{name}: gradient norm: {param.grad.norm()}")
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
            self.optimizer.step()

        # print(f"Train loss: {l.detach().cpu().numpy()} - accuracy: {acc}")

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        all_crops = batch["all_crops"]
        NREF = batch["reference_crop_index"]

        all_crops = all_crops.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with (
                autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                all_crop_embeddings = self.network(all_crops)
                # This rearrange below isn't necessary, but would come in handy when doing more involved contrastive tasks.
                # z_i_embeddings = rearrange(
                #     all_crop_embeddings[:NREF],
                #     "(b NREF) c -> b NREF c",
                #     b=self.batch_size,
                # )
                # z_j_embeddings = rearrange(
                #     all_crop_embeddings[NREF:],
                #     "(b NREF) c -> b NREF c",
                #     b=self.batch_size,
                # )

                # Normalize prior to contrastive loss
                z_i_embeddings = nn.functional.normalize(
                    all_crop_embeddings[:NREF], dim=1
                )
                z_j_embeddings = nn.functional.normalize(
                    all_crop_embeddings[NREF:], dim=1
                )

                # del data
                l, acc = self.loss(z_i_embeddings, z_j_embeddings)
                # print(f"Val loss: {l.detach().cpu().numpy()} - accuracy: {acc}")

        return {"loss": l.detach().cpu().numpy()}


####################################################################
############################# VARIANTS #############################
####################################################################


class SimCLRTrainer_BS6(SimCLRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 6


class SimCLRTrainer_BS32(SimCLRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 32


class SimCLRTrainer_warmup50ep(SimCLRTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):

        super(SimCLRTrainer_warmup50ep, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )

        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)
        plan.configurations[configuration_name].patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

    def configure_optimizers(self, stage: str = "warmup_all"):
        """
        Two-stage training:
        1) warmup_all  → linear LR warmup for `warmup_duration_whole_net` epochs
        2) train       → poly LR decay starting AFTER warmup
        """
        assert stage in ["warmup_all", "train"]

        # If already in this stage, return existing schedulers
        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        # Select parameters (DDP-safe)
        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        # -------------------------------
        # 1) WARMUP STAGE
        # -------------------------------
        if stage == "warmup_all":
            self.print_to_log_file("train whole net → WARMUP stage")

            # fresh optimizer
            optimizer = torch.optim.AdamW(
                params,
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.98),
                amsgrad=False,
                fused=True,
            )

            # linear warmup → reaches initial_lr at warmup_duration_whole_net
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer,
                max_lr=self.initial_lr,
                max_steps=self.warmup_duration_whole_net,
            )

            self.print_to_log_file(
                f"[Warmup] Initialized at epoch {self.current_epoch}"
            )

        # -------------------------------
        # 2) TRAIN STAGE (after warmup)
        # -------------------------------
        else:
            self.print_to_log_file("train whole net → TRAIN stage")

            # If transitioning warmup → train:
            if self.training_stage == "warmup_all":
                # keep optimizer from warmup (preserve momentum)
                optimizer = self.optimizer
                self.print_to_log_file(
                    "Reusing optimizer from warmup (momentum preserved)."
                )
            else:
                # If train is called directly (no warmup)
                optimizer = torch.optim.AdamW(
                    params,
                    lr=self.initial_lr,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.98),
                    amsgrad=False,
                    fused=True,
                )

            # Poly LR decay starting AFTER warmup duration
            lr_scheduler = PolyLRScheduler_offset(
                optimizer=optimizer,
                initial_lr=self.initial_lr,
                max_steps=self.num_epochs,
                start_step=self.warmup_duration_whole_net,
            )

            self.print_to_log_file(f"[Train] Initialized at epoch {self.current_epoch}")

        # Update state
        self.training_stage = stage
        empty_cache(self.device)

        # Store inside object so next call knows what's already set
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        super().on_train_epoch_start()

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith(
                "module."
            ):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]

        self.current_epoch = checkpoint["current_epoch"]
        min_epoch = self.logger.load_checkpoint(checkpoint["logging"])
        # Apparently the val log is not written correctly when we currently save the checkpoint.
        self.current_epoch = min_epoch
        self._best_ema = checkpoint["_best_ema"]

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
        # and lr scheduler are already set up
        if self.current_epoch < self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


class SimCLRTrainer_warmup50ep_BS8(SimCLRTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):

        super(SimCLRTrainer_warmup50ep_BS8, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )

        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)
        plan.configurations[configuration_name].patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        self.total_batch_size = 8
