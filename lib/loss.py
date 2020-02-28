import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from lib.knn.__init__ import KNearestNeighbor


def loss_calculation(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter, rot_anchors, sym_list):
    """
    Args:
        pred_t: bs x num_p x 3
        target_t: bs x num_p x 3
        pred_r: bs x num_rot x 4
        pred_c: bs x num_rot
        target_r: bs x num_point_mesh x 3
        rot_anchors: num_rot x 4
        model_points: bs x num_point_mesh x 3
        idx: bs x 1, index of object in object class list
    Return:
        loss:

    """
    knn = KNearestNeighbor(1)
    bs, num_p, _ = pred_t.size()
    num_rot = pred_r.size()[1]
    num_point_mesh = model_points.size()[1]
    # regularization loss
    rot_anchors = torch.from_numpy(rot_anchors).float().cuda()
    rot_anchors = rot_anchors.unsqueeze(0).repeat(bs, 1, 1).permute(0, 2, 1)
    cos_dist = torch.bmm(pred_r, rot_anchors)   # bs x num_rot x num_rot
    loss_reg = F.threshold((torch.max(cos_dist, 2)[0] - torch.diagonal(cos_dist, dim1=1, dim2=2)), 0.001, 0)
    loss_reg = torch.mean(loss_reg)
    # rotation loss
    rotations = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_rot, 1),\
                           (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_rot, 1), \
                           (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_rot, 1), \
                           (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
                           (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_rot, 1)), dim=2).contiguous().view(bs*num_rot, 3, 3)
    rotations = rotations.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_rot, 1, 1).view(bs*num_rot, num_point_mesh, 3)
    pred_r = torch.bmm(model_points, rotations)
    if idx[0].item() in sym_list:
        target_r = target_r[0].transpose(1, 0).contiguous().view(3, -1)
        pred_r = pred_r.permute(2, 0, 1).contiguous().view(3, -1)
        inds = knn(target_r.unsqueeze(0), pred_r.unsqueeze(0))
        target_r = torch.index_select(target_r, 1, inds.view(-1).detach() - 1)
        target_r = target_r.view(3, bs*num_rot, num_point_mesh).permute(1, 2, 0).contiguous()
        pred_r = pred_r.view(3, bs*num_rot, num_point_mesh).permute(1, 2, 0).contiguous()
    dis = torch.mean(torch.norm((pred_r - target_r), dim=2), dim=1)
    dis = dis / obj_diameter   # normalize by diameter
    pred_c = pred_c.contiguous().view(bs*num_rot)
    loss_r = torch.mean(dis / pred_c + torch.log(pred_c), dim=0)
    # translation loss
    loss_t = F.smooth_l1_loss(pred_t, target_t, reduction='mean')
    # total loss
    loss = loss_r + 2.0 * loss_reg + 5.0 * loss_t
    del knn
    return loss, loss_r, loss_t, loss_reg


class Loss(_Loss):
    def __init__(self, sym_list, rot_anchors):
        super(Loss, self).__init__(True)
        self.sym_list = sym_list
        self.rot_anchors = rot_anchors

    def forward(self, pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter):
        """
        """
        return loss_calculation(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter, self.rot_anchors, self.sym_list)
