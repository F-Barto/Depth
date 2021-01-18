import torch

from losses.elements.supervised_loss import SupervisedLoss
from losses.loss_base import LossBase

class HintedLoss(LossBase):
    def __init__(self, supervised_method='reprojected', hinted_loss_weight=1.0, supervised_num_scales=4):
        super().__init__()

        self.hinted_loss_weight = hinted_loss_weight

        self._supervised_loss = SupervisedLoss(supervised_method=supervised_method,
                                               supervised_num_scales=supervised_num_scales)
        self.supervised_loss = self._supervised_loss.calculate_losses

        self.supervised_method = supervised_method

        self.n = supervised_num_scales


    def calc_depth_hints_mask(self, photometric_losses, gt_photometric_losses):
        depth_hints_masks = []

        len_photo = len(photometric_losses[0]) # the length is the same for all scales, hence just [0]
        len_gt_photo = len(gt_photometric_losses[0])

        for i in range(self.n):
            all_losses = torch.cat(photometric_losses[i] + gt_photometric_losses[i], dim=1)

            # we keep all
            idxs = torch.argmin(all_losses, dim=1, keepdim=True).detach()

            # compute mask for each source views
            depth_hint_mask = []
            for i in range(len_photo, len_photo+len_gt_photo):
                depth_hint_mask.append((idxs == 2))
            depth_hint_mask = torch.cat(depth_hint_mask, dim=1)

            # if, in any source view, depth hint reprojection better than estimated or identity reprojection keep it
            depth_hint_mask = depth_hint_mask.any(dim=1, keepdim=True)

            depth_hints_masks.append(depth_hint_mask)

        return depth_hints_masks


    def calc_depth_hints_loss(self, depth_hints_masks, inv_depths, gt_depths):

        supervised_losses = self.supervised_loss(inv_depths, gt_depths, valid_masks=depth_hints_masks)

        depth_hints_loss = sum([supervised_losses[i].mean() for i in range(self.n)]) / self.n

        # Store and return reduced photometric loss
        self.add_metric('depth_hints_loss', depth_hints_loss)
        return depth_hints_loss

    def forward(self, photometric_losses, gt_photometric_losses, inv_depths, gt_depths):

        depth_hints_mask = self.calc_depth_hints_mask(photometric_losses, gt_photometric_losses)

        depth_hints_loss = self.calc_depth_hints_loss(depth_hints_mask, inv_depths, gt_depths)
        depth_hints_loss = self.hinted_loss_weight * depth_hints_loss

        return depth_hints_loss