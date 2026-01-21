import os
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from deprecated import deprecated
from typing_extensions import override
from dataclasses import asdict
from torch._dynamo import OptimizedModule
from nnssl.utilities.helpers import empty_cache

import torch
import torch.nn.functional as F
from nnssl.adaptation_planning.adaptation_plan import (
    AdaptationPlan,
    ArchitecturePlans,
    DynamicArchitecturePlans,
)
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.get_network_from_plan import get_network_from_plans
from nnssl.data.nnsslFilter.iqs_filter import OpenMindIQSFilter
from nnssl.data.nnsslFilter.modality_filter import ModalityFilter
from nnssl.data.raw_dataset import Collection
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.ssl_data.data_augmentation.transforms_for_dummy_2d import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
)
from nnssl.ssl_data.dataloading.data_loader_3d import (
    nnsslIndexableCenterCropDataLoader3D,
)
from nnssl.ssl_data.dataloading.indexable_dataloader import (
    IndexableSingleThreadedAugmenter,
)
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper

from nnssl.training.loss.mse_loss import (
    MAEMSELoss,
    LossMaskMSELoss,
    AnatDistWeightedMAEMSELoss,
    AnatWeightedMAEMSELoss,
    AnatDistExpWeightedMAEMSELoss,
    AnatDistGaussWeightedMAEMSELoss,
)
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from torch import nn
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform,
    MirrorTransform,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import join
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import numpy as np
from nnssl.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
)
from batchgenerators.utilities.file_and_folder_operations import load_json

def create_blocky_mask(
    tensor_size, block_size, sparsity_factor=0.75, rng_seed: None | int = None
) -> torch.Tensor:
    """
    Create the smallest binary mask for the encoder by choosing a percentage of pixels at that resolution..

    :param tensor_size: Tuple of the dimensions of the tensor (height, width, depth).
    :param block_size: Size of the block to be masked (set to 0) in the smaller mask.
    :return: A binary mask tensor.
    """
    # Calculate the size of the smaller mask
    small_mask_size = tuple(size // block_size for size in tensor_size)

    # Create the smaller mask
    flat_mask = torch.ones(np.prod(small_mask_size))
    n_masked = int(sparsity_factor * flat_mask.shape[0])
    if rng_seed is None:
        mask_indices = torch.randperm(flat_mask.shape[0])[:n_masked]
    else:
        gen = torch.Generator.manual_seed(rng_seed)
        mask_indices = torch.randperm(flat_mask.shape[0], generator=gen)[:n_masked]
    flat_mask[mask_indices] = 0
    small_mask = torch.reshape(flat_mask, small_mask_size)
    return small_mask


class BaseMAETrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (160, 160, 160)
        self.mask_percentage: float = 0.75

        self.im_output_folder = os.path.join(self.output_folder, "img_log")
        os.makedirs(self.im_output_folder, exist_ok=True)
        self.save_imgs_every_n_epochs = 200

    def initialize(self):
        # self.recon_dataloader = self.get_qual_recon_dataloader()
        super(BaseMAETrainer, self).initialize()

    @staticmethod
    def mask_creation(
        batch_size: int,
        patch_size: tuple[int, int, int],
        mask_percentage: float,
        rng_seed: int | None = None,
        block_size: int = 16,
    ) -> torch.Tensor:
        """
        Creates a masking tensor with 1s (indicating no masking) and 0s (indicating masking).
        The mask has to be of same size like the input data (batch_size, 1, x, y, z).

        :param batch_size: batch size during training
        :param patch_size: The 3D shape information for the input patch.
        :param mask_percentage: percentage of the patch that should be masked
        :param block_size: size of the blocks that should be masked
        :return:
        """

        sparsity_factor = mask_percentage
        mask = [
            create_blocky_mask(patch_size, block_size, sparsity_factor)
            for _ in range(batch_size)
        ]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return MAEMSELoss()

    @override
    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> nn.Module:
        # ---------------------------- Create architecture --------------------------- #
        architecture = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        arch_plans = ArchitecturePlans(arch_class_name="ResEncL")
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plans,
            pretrain_plan=self.plan,
            pretrain_num_input_channels=num_input_channels,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=(
                "encoder.stem.convs.0.conv",
                "encoder.stem.convs.0.all_modules.0",
            ),
        )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        return architecture, adapt_plan

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically.
        """
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
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size)

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

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

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
            output = self.network(masked_data)
            # del data
            l = self.loss(output, data, mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(masked_data)
            l = self.loss(output, data, mask)

        return {"loss": l.detach().cpu().numpy()}

    @deprecated
    @staticmethod
    def rescale_images(
        img_arr: torch.Tensor,
        recon_arr: torch.Tensor,
        full_img_min: float,
        full_img_max: float,
    ) -> np.ndarray:
        img_arr = (img_arr - full_img_min) / (full_img_max - full_img_min)
        rec_arr = (recon_arr - full_img_min) / (full_img_max - full_img_min)
        return img_arr, rec_arr

    def log_img_volume(
        self,
        img: np.ndarray | torch.Tensor,
        meta_info: dict,
        filename: str,
        dtype: np.dtype = np.float32,
    ):
        """Logs a 3D numpy array given the meta info to output folder with filename for visual inspection"""
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = img.squeeze().astype(dtype)
        sitk_img: sitk.Image = sitk.GetImageFromArray(img)
        sitk_img.SetSpacing(meta_info["sitk_stuff"]["spacing"])
        sitk_img.SetOrigin(meta_info["sitk_stuff"]["origin"])
        sitk_img.SetDirection(meta_info["sitk_stuff"]["direction"])
        sitk.WriteImage(sitk_img, os.path.join(self.im_output_folder, filename))

    def get_qual_recon_dataloader(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()
        dl_val = self.get_centercrop_val_dataloader()

        mt_gen_val = IndexableSingleThreadedAugmenter(dl_val, val_transforms)
        return mt_gen_val

    def get_centercrop_val_dataloader(self):
        """Returns a centercropped dataloader."""
        _, dataset_val = self.get_tr_and_val_datasets()

        dl_val = nnsslIndexableCenterCropDataLoader3D(
            dataset_val,
            1,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
            max_samples=25,
        )
        return dl_val

    def run_training(self):
        try:
            self.on_train_start()
            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []
                for batch_id in tqdm(
                    range(self.num_iterations_per_epoch),
                    desc=f"Epoch {epoch}",
                    disable=(
                        True
                        if (
                            ("LSF_JOBID" in os.environ)
                            or ("SLURM_JOB_ID" in os.environ)
                        )
                        else False
                    ),
                ):
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_batch = next(self.dataloader_val)
                        val_outputs.append(self.validation_step(val_batch))
                    self.on_validation_epoch_end(val_outputs)

                self.on_epoch_end()
                if self.exit_training_flag:
                    print("Finished last epoch before restart.")
                    self.print_to_log_file("Finished last epoch before restart.")
                    raise KeyboardInterrupt

            self.on_train_end()
        except KeyboardInterrupt:
            self.print_to_log_file("Keyboard interrupt. Exiting gracefully.")
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            raise KeyboardInterrupt

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=rotation_for_DA["x"],
                angle_y=rotation_for_DA["y"],
                angle_z=rotation_for_DA["z"],
                p_rot_per_axis=1,  # todo experiment with this
                do_scale=True,
                scale=(0.7, 1.4),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=order_resampling_data,
                # ToDo: Why do we even do scale transforms and do specifically preprocess data? This largely makes no sense, right?
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_resampling_seg,
                random_crop=False,  # random cropping is part of our dataloaders
                p_el_per_sample=0,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False,  # todo experiment with this
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(NumpyToTensor(["data"], "float"))
        tr_transforms.append(NumpyToTensor(["seg"], "long"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        val_transforms = []
        val_transforms.append(NumpyToTensor(["data"], "float"))
        val_transforms.append(NumpyToTensor(["seg"], "long"))
        val_transforms = Compose(val_transforms)
        return val_transforms


####################################################################
############################# VARIANTS #############################
####################################################################


############################# WARMUP 50 EPOCHS #############################
class BaseMAETrainer_warmup50ep(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_warmup50ep, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

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


############################# ANON & ANAT BASE CLASSES #############################


class BaseMAETrainer_ANAT(BaseMAETrainer):

    def get_dataloaders(self):
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
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_foreground_dataloaders(initial_patch_size)

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


class BaseMAETrainer_weightedANAT(BaseMAETrainer):

    def get_dataloaders(self):
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
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_dist_dataloaders(initial_patch_size)

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

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistWeightedMAEMSELoss()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        dist = data[:, -1].to(self.device, non_blocking=True)
        dist = dist.unsqueeze(1)
        data = data[:, 0].to(self.device, non_blocking=True)
        data = data.unsqueeze(1)
        anat = batch["seg"].to(self.device, non_blocking=True)

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

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
            output = self.network(masked_data)
            # del data
            l = self.loss(output, data, anat, dist, mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        dist = data[:, -1].to(self.device, non_blocking=True)
        dist = dist.unsqueeze(1)
        data = data[:, 0].to(self.device, non_blocking=True)
        data = data.unsqueeze(1)
        anat = batch["seg"].to(self.device, non_blocking=True)

        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(masked_data)
            l = self.loss(output, data, anat, dist, mask)

        return {"loss": l.detach().cpu().numpy()}


class BaseMAETrainer_weightedANAT_ExpWeight_BS8(BaseMAETrainer_weightedANAT):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.loss_alpha = 0.5

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistExpWeightedMAEMSELoss(alpha=self.loss_alpha)


class BaseMAETrainer_weightedANAT_GaussWeight_BS8(BaseMAETrainer_weightedANAT):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.loss_sigma = 1.0

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistGaussWeightedMAEMSELoss(alpha=self.loss_sigma)


class BaseMAETrainer_weightedANAT_BS8(BaseMAETrainer_weightedANAT):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class BaseMAETrainer_weightedANAT_warmup50ep(BaseMAETrainer_weightedANAT):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_warmup50ep, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

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


class BaseMAETrainer_weightedANAT_warmup50ep_mask065(
    BaseMAETrainer_weightedANAT_warmup50ep
):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_warmup50ep_mask065, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        # masking percentage
        self.mask_percentage = 0.65


class BaseMAETrainer_weightedANAT_warmup50ep_mask055(
    BaseMAETrainer_weightedANAT_warmup50ep
):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_warmup50ep_mask055, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        # masking percentage
        self.mask_percentage = 0.55


class BaseMAETrainer_weightedANAT_warmup50ep_featLoss(BaseMAETrainer_weightedANAT_warmup50ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_warmup50ep_featLoss, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        self.plan = plan
        self.configuration_name = configuration_name
        self.device = device

    def build_teacher(self):
        assert self.plan["teacher"] is not None, "Teacher model is not defined"
        assert os.path.exists(self.plan["teacher"]), f"Teacher model folder '{self.plan["teacher"]}' does not exist"
        teacher_cfg_file = os.path.join(self.plan["teacher"], "plans.json")

        teacher_data_file = os.path.join(self.plan["teacher"], "dataset.json")

        # Load teacher configuration
        teacher_cfg = load_json(teacher_cfg_file)
        teacher_cfg = teacher_cfg["configurations"]["3d_fullres"]
        arch_cfg = teacher_cfg["architecture"]

        # Load teacher dataset information
        dataset_cfg = load_json(teacher_data_file)

        # Load teacher network
        teacher_network = get_network_from_plans(arch_class_name=arch_cfg["network_class_name"],
                                                 arch_kwargs=arch_cfg["arch_kwargs"],
                                                 arch_kwargs_req_import=arch_cfg["_kw_requires_import"],
                                                 input_channels=1,
                                                 output_channels=len(list(dataset_cfg["labels"].keys())),
                                                 allow_init=True,
                                                 deep_supervision=False).to(self.device)
        
        return teacher_network
    

    def load_teacher_checkpoint(self):
        ckpt_path = os.path.join(self.plan["teacher"],"fold_all","checkpoint_final.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint path '{ckpt_path}' does not exist"

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in ckpt["network_weights"].items()
        }

        self.teacher_network.load_state_dict(state_dict, strict=True)

        self.teacher_network.eval()
        for p in self.teacher_network.parameters():
            p.requires_grad = False

        return self.teacher_network


    def initialize(self):
        super(BaseMAETrainer_weightedANAT_warmup50ep_featLoss, self).initialize()

        # ----------------------------
        # Build & load teacher
        # ----------------------------
        self.teacher_network = self.build_teacher()
        self.teacher_network = self.load_teacher_checkpoint()
        self.print_to_log_file("Teacher loaded and frozen.")

        # ----------------------------
        # Feature buffers
        # ----------------------------
        self.student_features = {}
        self.teacher_features = {}

        def save_feature(store, name):
            def hook(_, __, out):
                store[name] = out
            return hook

        # ----------------------------
        # Hook deepest encoder block
        # ----------------------------
        student_net = self.network.module if self.is_ddp else self.network
        teacher_net = self.teacher_network

        # Student: get last block of last stage
        student_stage = student_net.encoder.stages[-1]
        if hasattr(student_stage, "blocks"):
            student_block = student_stage.blocks[-1]
        else:
            student_block = list(student_stage.children())[-1]

        # Teacher: get last block of last stage
        teacher_stage = teacher_net.encoder.stages[-1]
        if hasattr(teacher_stage, "blocks"):
            teacher_block = teacher_stage.blocks[-1]
        else:
            teacher_block = list(teacher_stage.children())[-1]

        student_layer_name = "student.encoder.stages[-1].blocks[-1]"
        teacher_layer_name = "teacher.encoder.stages[-1].blocks[-1]"

        student_block.register_forward_hook(save_feature(self.student_features, student_layer_name))
        teacher_block.register_forward_hook(save_feature(self.teacher_features, teacher_layer_name))

        # ----------------------------
        # Projection head (student -> teacher channels)
        # ----------------------------
        with torch.no_grad():
            dummy = torch.zeros(
                1, 1, *self.config_plan.patch_size, device=self.device
            )
            _ = student_net(dummy)
            _ = teacher_net(dummy)

            student_C = self.student_features[student_layer_name].shape[1]
            teacher_C = self.teacher_features[teacher_layer_name].shape[1]

            # clear buffers after dummy pass
            self.student_features.clear()
            self.teacher_features.clear()

        self.student_proj = nn.Sequential(
            nn.Conv3d(student_C, teacher_C, kernel_size=1, bias=False),
            nn.InstanceNorm3d(teacher_C, affine=False),
        ).to(self.device)

        # ----------------------------
        # Distillation config
        # ----------------------------
        self.target_frac = 0.1 
        self.print_to_log_file(
            "Distillation enabled"
            f"warmup = {self.warmup_duration_whole_net} epochs"
        )


    def train_step(self, batch: dict) -> dict:
        # ----------------------------
        # Clear feature buffers
        # ----------------------------
        self.student_features.clear()
        self.teacher_features.clear()

        # ----------------------------
        # Data loading
        # ----------------------------
        data = batch["data"]
        dist = data[:, -1].to(self.device, non_blocking=True).unsqueeze(1)
        data = data[:, 0].to(self.device, non_blocking=True).unsqueeze(1)
        anat = batch["seg"].to(self.device, non_blocking=True)

        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = mask.repeat_interleave(rep_D, dim=2).repeat_interleave(rep_H, dim=3).repeat_interleave(rep_W, dim=4)
        masked_data = data * mask

        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            _ = self.teacher_network(data)

        # ----------------------------
        # Forward pass
        # ----------------------------
        student_layer_name = "student.encoder.stages[-1].blocks[-1]"
        teacher_layer_name = "teacher.encoder.stages[-1].blocks[-1]"
        with autocast(self.device.type, enabled=(self.device.type=="cuda")) if self.device.type=="cuda" else dummy_context():
            # Student
            student_out = self.network(masked_data)
            recon_loss = self.loss(student_out, data, anat, dist, mask)

            # Feature distillation
            feat_loss = torch.tensor(0.0, device=self.device)
            lambda_dyn = 0.0

            if self.current_epoch >= self.warmup_duration_whole_net:
                student_feat = self.student_features[student_layer_name]
                teacher_feat = self.teacher_features[teacher_layer_name]

                # Resize teacher features to match student
                teacher_feat = F.interpolate(
                    teacher_feat,
                    size=student_feat.shape[2:],
                    mode="trilinear",
                    align_corners=False
                )

                student_feat = self.student_proj(student_feat)
                feat_loss = F.mse_loss(student_feat, teacher_feat)

                # Dynamical lambda balancing
                lambda_dyn = self.target_frac * recon_loss.detach() / (feat_loss.detach() + 1e-8)

            total_loss = recon_loss + lambda_dyn * feat_loss

        # ----------------------------
        # Backprop
        # ----------------------------
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "recon_loss": recon_loss.detach().cpu().numpy(),
            "feat_loss": feat_loss.detach().cpu().numpy()
        } 
        

class BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep_featLoss(
    BaseMAETrainer_weightedANAT_warmup50ep_featLoss
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep_featLoss, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        self.plan = plan
        self.configuration_name = configuration_name
        self.device = device

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistGaussWeightedMAEMSELoss(sigma=self.loss_sigma)


class BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep_featLoss(
    BaseMAETrainer_weightedANAT_warmup50ep_featLoss
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep_featLoss, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        self.plan = plan
        self.configuration_name = configuration_name
        self.device = device

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistExpWeightedMAEMSELoss(sigma=self.loss_sigma)

class BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep_BS8(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep_BS8, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

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
        self.loss_alpha = 0.5

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistExpWeightedMAEMSELoss(alpha=self.loss_alpha)


class BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None
        self.total_batch_size = 2
        self.loss_alpha = 0.5

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistExpWeightedMAEMSELoss(alpha=self.loss_alpha)


class BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep_mask065(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep_mask065, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None
        self.total_batch_size = 2
        self.loss_alpha = 0.5

        # mask percentage
        self.mask_percentage = 0.65

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistExpWeightedMAEMSELoss(alpha=self.loss_alpha)


class BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep_mask055(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_ExpWeight_warmup50ep_mask055, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None
        self.total_batch_size = 2
        self.loss_alpha = 0.5

        # mask percentage
        self.mask_percentage = 0.55

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistExpWeightedMAEMSELoss(alpha=self.loss_alpha)


class BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep_BS8(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep_BS8, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

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
        self.loss_sigma = 1.0

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistGaussWeightedMAEMSELoss(sigma=self.loss_sigma)


class BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None
        self.total_batch_size = 2
        self.loss_sigma = 1.0

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistGaussWeightedMAEMSELoss(sigma=self.loss_sigma)


class BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep_mask065(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(
            BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep_mask065, self
        ).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None
        self.total_batch_size = 2
        self.loss_alpha = 0.5

        # mask percentage
        self.mask_percentage = 0.65

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistGaussWeightedMAEMSELoss(alpha=self.loss_alpha)


class BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep_mask055(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(
            BaseMAETrainer_weightedANAT_GaussWeight_warmup50ep_mask055, self
        ).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None
        self.total_batch_size = 2
        self.loss_alpha = 0.5

        # mask percentage
        self.mask_percentage = 0.55

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return AnatDistGaussWeightedMAEMSELoss(alpha=self.loss_alpha)


class BaseMAETrainer_weightedANAT_warmup50ep_lr1e2_BS8(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_weightedANAT_warmup50ep_lr1e2_BS8, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 0.1
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None
        self.total_batch_size = 8


class BaseMAETrainer_weightedANAT_warmup50ep_BS8(
    BaseMAETrainer_weightedANAT_warmup50ep
):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class BaseMAETrainer_dilatedANAT(BaseMAETrainer):

    def get_dataloaders(self):
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
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_anatDilated_dataloaders(initial_patch_size)

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

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        _lambda = 10.0
        return AnatWeightedMAEMSELoss(_lambda=_lambda)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)
        anat = batch["seg"].to(self.device, non_blocking=True)

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

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
            output = self.network(masked_data)
            # del data
            l = self.loss(output, data, anat, mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)
        anat = batch["seg"].to(self.device, non_blocking=True)

        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(masked_data)
            l = self.loss(output, data, anat, mask)

        return {"loss": l.detach().cpu().numpy()}


class BaseMAETrainer_dilatedANAT_BS8(BaseMAETrainer_dilatedANAT):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class BaseMAETrainer_dilatedANAT_warmup50ep(BaseMAETrainer_dilatedANAT):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(BaseMAETrainer_dilatedANAT_warmup50ep, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )
        # Fix the input patch size
        self.config_plan.patch_size = (160, 160, 160)

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


class BaseMAETrainer_dilatedANAT_warmup50ep_BS8(BaseMAETrainer_dilatedANAT_warmup50ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class BaseMAETrainer_ANON(BaseMAETrainer):

    def build_loss(self):
        return LossMaskMSELoss()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        anon = batch["seg"]
        data = data.to(self.device, non_blocking=True)
        anon = anon.to(self.device, non_blocking=True)

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask
        loss_mask = (1 - mask) * (1 - anon)

        self.optimizer.zero_grad(set_to_none=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(masked_data)
            # del data
            l = self.loss(output, data, loss_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        anon = batch["seg"]
        data = data.to(self.device, non_blocking=True)
        anon = anon.to(self.device, non_blocking=True)

        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask
        loss_mask = (1 - mask) * (1 - anon)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(masked_data)
            l = self.loss(output, data, loss_mask)

        return {"loss": l.detach().cpu().numpy()}


class BaseMAETrainer_ANAT_ANON(BaseMAETrainer_ANAT, BaseMAETrainer_ANON):
    pass


############################# BASELINE #############################


class BaseMAETrainer_BS8(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


############################# LOWER LR #############################


class BaseMAETrainer_lowlr(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.initial_lr = 0.001


############################# MASKS & IQS #############################


class BaseMAETrainer_ANAT_ANON_BS8(BaseMAETrainer_ANAT_ANON):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class BaseMAETrainer_BS8_IQS1_5(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.iimg_filters.append(
            OpenMindIQSFilter(Collection.from_dict(self.pretrain_json), 1.5)
        )


class BaseMAETrainer_BS8_IQS2_5(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.iimg_filters.append(
            OpenMindIQSFilter(Collection.from_dict(self.pretrain_json), 2.5)
        )


class BaseMAETrainer_BS8_IQS3_0(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.iimg_filters.append(
            OpenMindIQSFilter(Collection.from_dict(self.pretrain_json), 3.0)
        )


class BaseMAETrainer_BS8_T1w_T2w_FLAIR(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.iimg_filters.append(
            ModalityFilter(valid_modalities=["T1w", "T2w", "FLAIR"])
        )


class BaseMAETrainer_BS8_IQS3_5_FA(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.iimg_filters.extend(
            [
                ModalityFilter(valid_modalities=["FA"]),
                OpenMindIQSFilter(Collection.from_dict(self.pretrain_json), 3.5),
            ]
        )
        self.num_val_iterations_per_epoch = 5


############################# OTHERS #############################


class BaseMAETrainer_BS8_100ep(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.num_epochs = 100


class BaseMAETrainer_BS1(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 1
        self.num_epochs = 1000


class BaseMAETrainer_BS2(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2
        self.num_epochs = 1000


class BaseMAETrainer_BS8_1000ep(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.num_epochs = 1000


############################# TESTING #############################


class BaseMAETrainer_Test(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (96, 96, 96)
        assert self.plan.configurations[configuration_name].patch_size == (
            96,
            96,
            96,
        ), "Patch size not preserved to downsteam"
        self.total_batch_size = 2
        self.num_epochs = 3


class BaseMAETrainer_Test_defaultpatch(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (160, 160, 160)
        assert self.plan.configurations[configuration_name].patch_size == (
            160,
            160,
            160,
        ), "Patch size not preserved to downsteam"
        self.total_batch_size = 2
        self.num_epochs = 3


class BaseMAETrainer_ANAT_ANON_test(BaseMAETrainer_ANAT_ANON):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.plan.configurations[configuration_name].patch_size = (128, 128, 128)
        self.total_batch_size = 2


class BaseMAETrainer_BS8_IQS_test(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.iimg_filter = OpenMindIQSFilter(
            Collection.from_dict(self.pretrain_json), 2.5
        )
        self.total_batch_size = 1


class NonResEncL_BaseMAETrainer_Test(BaseMAETrainer_Test):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.architecture_kwargs: DynamicArchitecturePlans = DynamicArchitecturePlans(
            **{
                "n_stages": 6,
                "features_per_stage": [32, 64, 128, 256, 512, 512],
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "kernel_sizes": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
                "strides": [
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ],
                "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
                "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
                "conv_bias": True,
                "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU",
                "nonlin_kwargs": {"inplace": True},
            }
        )

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels, num_output_channels
    ):
        architecture = get_network_from_plans(
            arch_class_name="ResidualEncoderUNet",
            arch_kwargs=asdict(self.architecture_kwargs),
            arch_kwargs_req_import=["conv_op", "norm_op", "nonlin"],
            input_channels=num_input_channels,
            output_channels=num_output_channels,
            deep_supervision=False,
        )
        arch_plans = ArchitecturePlans(
            arch_class_name="ResidualEncoderUNet", arch_kwargs=self.architecture_kwargs
        )
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plans,
            pretrain_plan=self.plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=(
                "encoder.stem.convs.0.conv",
                "encoder.stem.convs.0.all_modules.0",
            ),
        )
        return architecture, adapt_plan


# TRAINERS RUNNING FOR 1,500 EPOCHS


class BaseMAETrainer_dilatedANAT_1500ep(BaseMAETrainer_dilatedANAT):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1500


class BaseMAETrainer_dilatedANAT_1500ep_BS8(BaseMAETrainer_dilatedANAT):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1500
        self.total_batch_size = 8


class BaseMAETrainer_weightedANAT_1500ep(BaseMAETrainer_weightedANAT):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1500


class BaseMAETrainer_weightedANAT_1500ep_BS8(BaseMAETrainer_weightedANAT):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1500
        self.total_batch_size = 8
