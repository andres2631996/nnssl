import torch

from nnssl.training.loss.abstract_loss import AbstractLoss
from nnssl.training.loss.compound_losses import DC_and_CE_loss
from nnssl.training.loss.mse_loss import MAEMSELoss


class DualDecoderMAELoss(AbstractLoss):
    """
    Reconstruction (MSE on masked voxels, as in MAEMSELoss) + segmentation
    (Dice+CE against an auxiliary segmentation target, e.g. a predicted vessel
    mask) for a dual-decoder MAE.
    """

    def __init__(
        self,
        seg_loss_weight: float = 1.0,
        batch_dice: bool = True,
        is_ddp: bool = False,
    ):
        super().__init__()
        self.recon = MAEMSELoss()
        self.seg = DC_and_CE_loss(
            soft_dice_kwargs={"batch_dice": batch_dice, "do_bg": False, "smooth": 1e-5, "ddp": is_ddp},
            ce_kwargs={},
        )
        self.seg_loss_weight = seg_loss_weight

    def forward(
        self,
        recon_out: torch.Tensor,
        seg_out: torch.Tensor,
        data: torch.Tensor,
        seg: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        l_recon = self.recon(recon_out, data, mask)
        l_seg = self.seg(seg_out, seg)
        return l_recon + self.seg_loss_weight * l_seg
