from typing import override, Union
import torch
from torch import nn

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.model_genesis_transform import ModelGenesisTransform
from nnssl.training.loss.mse_loss import LossMaskMSELoss
from nnssl.data.nnsslFilter.modality_filter import ModalityFilter

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import save_json

from nnssl.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
)
from torch._dynamo import OptimizedModule
from nnssl.utilities.helpers import empty_cache


class ModelGenesisTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):
        super(ModelGenesisTrainer, self).__init__(
            plan, configuration_name, fold, pretrain_json, device
        )
        self.config_plan.patch_size = (160, 160, 160)

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return torch.nn.MSELoss()

    @override
    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> nn.Module:
        network = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
        )
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=(
                "encoder.stem.convs.0.conv",
                "encoder.stem.convs.0.all_modules.0",
            ),
        )
        return network, adapt_plan

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically.
        """
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size

        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size=patch_size)

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
        in_data = batch["input"]
        target = batch["target"]
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

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
            output = self.network(in_data)
            # del data
            l = self.loss(target, output)

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
        in_data = batch["input"]
        target = batch["target"]
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with (
                autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                output = self.network(in_data)
                # del data
                l = self.loss(target, output)

        return {"loss": l.detach().cpu().numpy()}

    @staticmethod
    def get_training_transforms() -> AbstractTransform:
        tr_transforms = []

        tr_transforms.append(ModelGenesisTransform())
        tr_transforms.append(NumpyToTensor(["input", "target"], "float"))
        tr_transforms.append(NumpyToTensor(["seg"], "long"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        return ModelGenesisTrainer.get_training_transforms()


class ModelGenesisTrainer_warmup50ep(ModelGenesisTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(ModelGenesisTrainer_warmup50ep, self).__init__(
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


class ModelGenesisTrainer_BS8(ModelGenesisTrainer):

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


class ModelGenesisTrainer_warmup50ep_BS8(ModelGenesisTrainer_warmup50ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):

        super(ModelGenesisTrainer_warmup50ep_BS8, self).__init__(
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


class ModelGenesisTrainer_ANAT(ModelGenesisTrainer):

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically.
        """
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size

        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_foreground_dataloaders(initial_patch_size=patch_size)

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


class ModelGenesisTrainer_ANON(ModelGenesisTrainer):

    def build_loss(self):
        return LossMaskMSELoss()

    def train_step(self, batch: dict) -> dict:
        in_data = batch["input"]
        target = batch["target"]
        anon = batch["seg"]
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        anon = anon.to(self.device, non_blocking=True)

        loss_mask = 1 - anon

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(in_data)
            l = self.loss(output, target, loss_mask)

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
        in_data = batch["input"]
        target = batch["target"]
        anon = batch["seg"]
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        anon = anon.to(self.device, non_blocking=True)

        loss_mask = 1 - anon

        with torch.no_grad():
            with (
                autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                output = self.network(in_data)
                l = self.loss(output, target, loss_mask)

        return {"loss": l.detach().cpu().numpy()}


class ModelGenesisTrainer_ANAT_ANON(ModelGenesisTrainer_ANAT, ModelGenesisTrainer_ANON):
    pass


class ModelGenesisTrainer_ANAT_ANON_test(ModelGenesisTrainer_ANAT_ANON):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        plan.configurations[configuration_name].patch_size = (96, 96, 96)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class ModelGenesisTrainer_ANAT_ANON_BS8(ModelGenesisTrainer_ANAT_ANON):
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


class ModelGenesisTrainer_ANAT_ANON_BS8_T1w_T2w_FLAIR(ModelGenesisTrainer_ANAT_ANON):
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
        self.iimg_filters.append(
            ModalityFilter(valid_modalities=["T1w", "T2w", "FLAIR"])
        )
