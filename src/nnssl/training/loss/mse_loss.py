from torch import nn
import torch
from nnssl.training.loss.abstract_loss import AbstractLoss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from einops import rearrange
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt


class MAEMSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs"""
        # Mask = 1 represents not masked points
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)
        reconstruction_loss = torch.sum(reconstruction_loss * (1 - mask)) / torch.sum(
            (1 - mask)
        )

        return reconstruction_loss


class AnatDistWeightedMAEMSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        anat_mask: torch.Tensor,
        dist_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs"""
        # Mask = 1 represents not masked points
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)

        """
        plt.figure()
        plt.subplot(151)
        plt.imshow(
            target[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(152)
        plt.imshow(
            model_output[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(153)
        plt.imshow(
            dist_map[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(154)
        plt.imshow(
            anat_mask[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(155)
        plt.imshow(
            mask[0, 0, dist_map.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.savefig("/home/a870a/test_rec.png")
        plt.close()
        sys.exit()

        # Derive distance maps
        
        dist_map = torch.stack(
            [
                torch.from_numpy(
                    ndi.distance_transform_edt(
                        (anat_mask[b, 0].detach().cpu().numpy() < 1)
                    )
                )
                for b in range(target.shape[0])
            ],
            dim=0,
        ).to(
            model_output.device, dtype=model_output.dtype
        )  # (B, D, H, W)
        dist_map = dist_map.unsqueeze(1)
        """

        # Weighting: 1 inside vessels, 1 / (dist_map + eps) outside vessels
        weights = torch.where(
            anat_mask > 0.5,
            torch.ones_like(reconstruction_loss),  # original MSE inside vessels
            1.0 / (dist_map + 1.0),  # decreased MSE outside vessels
        )

        # Mask weights with mask from MAE
        effective_weights = weights * (1 - mask)

        reconstruction_loss = torch.sum(reconstruction_loss * effective_weights) / (
            torch.sum((effective_weights)) + 1e-5
        )
        return reconstruction_loss


class AnatDistWeightedMAEMSELoss_recDistMap(AbstractLoss):
    # Loss function combining penalized vasculature reconstruction
    # with reconstruction of vasculature distance map
    def __init__(self, _lambda=0.001):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")
        self._lambda = _lambda

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        anat_mask: torch.Tensor,
        dist_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs"""
        # Mask = 1 represents not masked points
        rec_output = model_output[0].unsqueeze(0)
        distmap_output = model_output[-1].unsqueeze(0)
        reconstruction_loss = (rec_output - target) ** 2  # (B, X, Y, Z, C)
        distmap_loss = (distmap_output - dist_map) ** 2

        distmap_loss = torch.sum(distmap_loss * (1 - mask)) / torch.sum((1 - mask))

        """
        plt.figure()
        plt.subplot(151)
        plt.imshow(
            target[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(152)
        plt.imshow(
            model_output[1, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(153)
        plt.imshow(
            dist_map[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(154)
        plt.imshow(
            anat_mask[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(155)
        plt.imshow(
            mask[0, 0, dist_map.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.savefig("/home/a870a/test_rec.png")
        plt.close()
        sys.exit()

        # Derive distance maps

        dist_map = torch.stack(
            [
                torch.from_numpy(
                    ndi.distance_transform_edt(
                        (anat_mask[b, 0].detach().cpu().numpy() < 1)
                    )
                )
                for b in range(target.shape[0])
            ],
            dim=0,
        ).to(
            model_output.device, dtype=model_output.dtype
        )  # (B, D, H, W)
        dist_map = dist_map.unsqueeze(1)
        """

        # Weighting: 1 inside vessels, 1 / (dist_map + eps) outside vessels
        weights = torch.where(
            anat_mask > 0.5,
            torch.ones_like(reconstruction_loss),  # original MSE inside vessels
            1.0 / (dist_map + 1.0),  # decreased MSE outside vessels
        )

        # Mask weights with mask from MAE
        effective_weights = weights * (1 - mask)

        reconstruction_loss = torch.sum(reconstruction_loss * effective_weights) / (
            torch.sum((effective_weights)) + 1e-5
        )

        reconstruction_loss += distmap_loss

        return reconstruction_loss


class MAEMSELoss_recDistMap(AbstractLoss):
    # Loss function combining penalized vasculature reconstruction
    # with reconstruction of vasculature distance map
    def __init__(self, _lambda=0.001):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")
        self._lambda = _lambda

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        dist_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs"""
        # Mask = 1 represents not masked points
        rec_output = model_output[0].unsqueeze(0)
        distmap_output = model_output[-1].unsqueeze(0)
        reconstruction_loss = (rec_output - target) ** 2  # (B, X, Y, Z, C)
        distmap_loss = (distmap_output - dist_map) ** 2

        reconstruction_loss = torch.sum(reconstruction_loss * (1 - mask)) / torch.sum(
            (1 - mask)
        )
        distmap_loss = torch.sum(distmap_loss * (1 - mask)) / torch.sum((1 - mask))
        """
        plt.figure()
        plt.subplot(151)
        plt.imshow(
            target[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(152)
        plt.imshow(
            model_output[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(153)
        plt.imshow(
            dist_map[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(154)
        plt.imshow(
            anat_mask[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(155)
        plt.imshow(
            mask[0, 0, dist_map.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.savefig("/home/a870a/test_rec.png")
        plt.close()
        sys.exit()

        # Derive distance maps
        
        dist_map = torch.stack(
            [
                torch.from_numpy(
                    ndi.distance_transform_edt(
                        (anat_mask[b, 0].detach().cpu().numpy() < 1)
                    )
                )
                for b in range(target.shape[0])
            ],
            dim=0,
        ).to(
            model_output.device, dtype=model_output.dtype
        )  # (B, D, H, W)
        dist_map = dist_map.unsqueeze(1)
        """
        reconstruction_loss += self._lambda * distmap_loss

        return reconstruction_loss


class AnatDistExpWeightedMAEMSELoss(AbstractLoss):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        anat_mask: torch.Tensor,
        dist_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        reconstruction_loss = (model_output - target) ** 2

        # Exponential decay outside anatomy
        outside_weights = torch.exp(-self.alpha * dist_map)

        weights = torch.where(
            anat_mask > 0.5,
            torch.ones_like(reconstruction_loss),  # inside anatomy
            outside_weights,  # outside anatomy
        )

        effective_weights = weights * (1 - mask)

        loss = torch.sum(reconstruction_loss * effective_weights) / (
            torch.sum(effective_weights) + 1e-5
        )

        return loss


class AnatDistGaussWeightedMAEMSELoss(AbstractLoss):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        anat_mask: torch.Tensor,
        dist_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        reconstruction_loss = (model_output - target) ** 2

        # Exponential decay outside anatomy
        outside_weights = torch.exp(-(dist_map**2) / (2 * self.sigma**2))

        weights = torch.where(
            anat_mask > 0.5,
            torch.ones_like(reconstruction_loss),  # inside anatomy
            outside_weights,  # outside anatomy
        )

        effective_weights = weights * (1 - mask)

        loss = torch.sum(reconstruction_loss * effective_weights) / (
            torch.sum(effective_weights) + 1e-5
        )

        return loss


class AnatWeightedMAEMSELoss(AbstractLoss):
    def __init__(self, _lambda=10.0):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")
        self._lambda = _lambda

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        anat_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs"""
        # Mask = 1 represents not masked points
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)

        # Weighting: 1 inside vessels, 1 / (dist_map + eps) outside vessels
        weights = torch.where(
            anat_mask > 0.5,
            torch.ones_like(reconstruction_loss)
            * self._lambda,  # original MSE inside vessels
            torch.ones_like(reconstruction_loss),  # decreased MSE outside vessels
        )

        # Mask weights with mask from MAE
        effective_weights = weights * (1 - mask)

        """
        plt.figure()
        plt.subplot(151)
        plt.imshow(
            target[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(152)
        plt.imshow(
            model_output[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(153)
        plt.imshow(
            effective_weights[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(154)
        plt.imshow(
            anat_mask[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.subplot(155)
        plt.imshow(
            mask[0, 0, model_output.shape[2] // 2].detach().cpu().numpy(),
            cmap="gray",
        )
        plt.colorbar()
        plt.savefig("/home/a870a/test_rec.png")
        plt.close()
        sys.exit()

        

        # Derive distance maps
        
        dist_map = torch.stack(
            [
                torch.from_numpy(
                    ndi.distance_transform_edt(
                        (anat_mask[b, 0].detach().cpu().numpy() < 1)
                    )
                )
                for b in range(target.shape[0])
            ],
            dim=0,
        ).to(
            model_output.device, dtype=model_output.dtype
        )  # (B, D, H, W)
        dist_map = dist_map.unsqueeze(1)
        """

        reconstruction_loss = torch.sum(reconstruction_loss * effective_weights) / (
            torch.sum((effective_weights)) + 1e-5
        )

        return reconstruction_loss


class AnatWeightedMAEMSELoss_lowerOutside(AbstractLoss):
    def __init__(self, _lambda=0.1):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")
        self._lambda = _lambda

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        anat_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs"""
        # Mask = 1 represents not masked points
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)

        # Weighting: 1 inside vessels, 1 / (dist_map + eps) outside vessels
        weights = torch.where(
            anat_mask > 0.5,
            torch.ones_like(reconstruction_loss),  # original MSE inside vessels
            torch.ones_like(reconstruction_loss)
            * self._lambda,  # decreased MSE outside vessels
        )

        # Mask weights with mask from MAE
        effective_weights = weights * (1 - mask)

        reconstruction_loss = torch.sum(reconstruction_loss * effective_weights) / (
            torch.sum((effective_weights)) + 1e-5
        )

        return reconstruction_loss


class LossMaskMSELoss(AbstractLoss):
    def forward(
        self, model_output: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """loss_mask = 1 in positions where loss calculation should take place"""
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)
        reconstruction_loss = torch.sum(reconstruction_loss * loss_mask) / torch.sum(
            loss_mask
        )

        return reconstruction_loss


class MAEL1Loss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="none")

    def forward(
        self,
        model_output: torch.Tensor,
        target: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        reconstruction_loss = torch.sum(
            torch.abs(model_output - target) * (1 - mask)
        ) / torch.sum((1 - mask))

        return reconstruction_loss


class MAESSIMLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        model_output: torch.Tensor,
        target: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        # Rescale output and target to [0, 1] for each image in batch
        output_min = torch.amin(model_output, dim=(1, 2, 3, 4), keepdim=True)
        output_max = torch.amax(model_output, dim=(1, 2, 3, 4), keepdim=True)
        target_min = torch.amin(target, dim=(1, 2, 3, 4), keepdim=True)
        target_max = torch.amax(target, dim=(1, 2, 3, 4), keepdim=True)

        rescaled_out = (model_output - output_min) / (output_max - output_min)
        rescaled_target = (target - target_min) / (target_max - target_min)

        # rescaled_out = rearrange(rescaled_out, "b x y z c -> b c x y z")
        # rescaled_target = rearrange(rescaled_target, "b x y z c -> b c x y z")

        ssim_loss = 1 - ssim(
            rescaled_out,
            rescaled_target,
            data_range=1,
            size_average=False,
            nonnegative_ssim=True,
        )

        return torch.mean(ssim_loss)


class MAESSIMLoss_WithMask(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        model_output: torch.Tensor,
        target: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        # Rescale output and target to [0, 1] for each image in batch
        output_min = torch.amin(model_output, dim=(1, 2, 3, 4), keepdim=True)
        output_max = torch.amax(model_output, dim=(1, 2, 3, 4), keepdim=True)
        target_min = torch.amin(target, dim=(1, 2, 3, 4), keepdim=True)
        target_max = torch.amax(target, dim=(1, 2, 3, 4), keepdim=True)

        rescaled_out = (model_output - output_min) / (output_max - output_min)
        rescaled_target = (target - target_min) / (target_max - target_min)

        # rescaled_out = rearrange(rescaled_out, "b x y z c -> b c x y z")
        # rescaled_target = rearrange(rescaled_target, "b x y z c -> b c x y z")
        rescaled_out = rescaled_out * (
            1 - mask
        )  # Make originally visible stuff 0, so that SSIM focuses on masked areas
        rescaled_target = rescaled_target * (1 - mask)

        ssim_loss = 1 - ssim(
            rescaled_out,
            rescaled_target,
            data_range=1,
            size_average=False,
            nonnegative_ssim=True,
        )

        return torch.mean(ssim_loss)


class MAE_MS_SSIMLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        model_output: torch.Tensor,
        target: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        # Rescale output and target to [0, 1] for each image in batch
        output_min = torch.amin(model_output, dim=(1, 2, 3, 4), keepdim=True)
        output_max = torch.amax(model_output, dim=(1, 2, 3, 4), keepdim=True)
        target_min = torch.amin(target, dim=(1, 2, 3, 4), keepdim=True)
        target_max = torch.amax(target, dim=(1, 2, 3, 4), keepdim=True)

        rescaled_out = (model_output - output_min) / (output_max - output_min)
        rescaled_target = (target - target_min) / (target_max - target_min)

        # rescaled_out = rearrange(rescaled_out, "b x y z c -> b c x y z")
        # rescaled_target = rearrange(rescaled_target, "b x y z c -> b c x y z")

        ssim_loss = 1 - ms_ssim(
            rescaled_out, rescaled_target, data_range=1, size_average=False, win_size=7
        )

        return torch.mean(ssim_loss)


class MAE_MS_SSIMLoss_WithMask(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        model_output: torch.Tensor,
        target: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        # Rescale output and target to [0, 1] for each image in batch
        output_min = torch.amin(model_output, dim=(1, 2, 3, 4), keepdim=True)
        output_max = torch.amax(model_output, dim=(1, 2, 3, 4), keepdim=True)
        target_min = torch.amin(target, dim=(1, 2, 3, 4), keepdim=True)
        target_max = torch.amax(target, dim=(1, 2, 3, 4), keepdim=True)

        rescaled_out = (model_output - output_min) / (output_max - output_min)
        rescaled_target = (target - target_min) / (target_max - target_min)

        # rescaled_out = rearrange(rescaled_out, "b x y z c -> b c x y z")
        # rescaled_target = rearrange(rescaled_target, "b x y z c -> b c x y z")

        rescaled_out = rescaled_out * (
            1 - mask
        )  # Set unmasked stuff 0, so MS SSIM focuses on masked areas
        rescaled_target = rescaled_target * (1 - mask)
        ssim_loss = 1 - ms_ssim(
            rescaled_out, rescaled_target, data_range=1, size_average=False, win_size=7
        )

        return torch.mean(ssim_loss)


class MSELoss_NoMask(AbstractLoss):
    def forward(
        self,
        model_output: torch.Tensor,
        target: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(model_output, target)

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="mean")


class L1Loss_NoMask(AbstractLoss):
    def forward(
        self,
        model_output: torch.Tensor,
        target: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(model_output, target)

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")
