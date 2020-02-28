import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pspnet import PSPNet
from lib.utils import sample_rotations_12, sample_rotations_24, sample_rotations_60


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):
    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()
        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class ModifiedDGCNN(nn.Module):
    def __init__(self, k):
        super(ModifiedDGCNN, self).__init__()
        self.k = k

        self.edge_conv1 = torch.nn.Conv2d(6, 64, 1)
        self.edge_conv2 = torch.nn.Conv2d(128, 64, 1)
        self.edge_conv3 = torch.nn.Conv2d(128, 128, 1)

        self.conv1 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv1_t = torch.nn.Conv1d(256, 256, 1)
        self.conv2_t = torch.nn.Conv1d(256, 1024, 1)

        self.conv1_r = torch.nn.Conv1d(256, 256, 1)
        self.conv2_r = torch.nn.Conv1d(256, 1024, 1)

    def get_nn_idx(self, x):
        """ Get nearest k neighbors.

        Args:
            x: bs x c x n_p

        Returns:
            nn_idx: bs x n_p x k

        """
        inner = torch.bmm(x.permute(0, 2, 1), x)
        square = torch.sum(torch.pow(x, 2), dim=1, keepdim=True)
        dist_mat = square.permute(0, 2, 1) + square - 2 * inner
        _, nn_idx = torch.topk(dist_mat, self.k, largest=False, dim=-1)
        return nn_idx

    def get_edge_feature(self, x, nn_idx):
        """ Construct edge feature.

        Args:
            x: bs x c x n_p
            nn_idx: bs x n_p x k

        Returns:
            edge_feature: bs x 2c x n_p x k

        """
        bs, c, n_p = x.size()
        nn_idx = torch.unsqueeze(nn_idx, 1).repeat(1, c, 1, 1).view(bs, c, n_p*self.k)
        neighbors = torch.gather(x, 2, nn_idx).view(bs, c, n_p, self.k)
        central = torch.unsqueeze(x, 3).repeat(1, 1, 1, self.k)
        edge_feature = torch.cat((central, neighbors - central), dim=1)
        return edge_feature

    def forward(self, x, emb):
        """ Extract point feature

        Args:
            x: bs x 3 x n_p
            emb: bs x c x n_p

        Returns:
            point_feat: bs x c_out x n_p
            global_feat: bs x 1024 x 1

        """
        np = x.size()[2]
        nn_idx = self.get_nn_idx(x)

        x = F.relu(self.edge_conv1(self.get_edge_feature(x, nn_idx)))
        x, _ = torch.max(x, dim=3, keepdim=False)
        x = F.relu(self.edge_conv2(self.get_edge_feature(x, nn_idx)))
        x, _ = torch.max(x, dim=3, keepdim=False)    # bs x 64 x n_p
        emb = F.relu(self.conv1(emb))
        point_feat = torch.cat((x, emb), dim=1)     # bs x 128 x n_p

        x = F.relu(self.edge_conv3(self.get_edge_feature(x, nn_idx)))
        x, _ = torch.max(x, dim=3, keepdim=False)
        emb = F.relu(self.conv2(emb))
        fusion = torch.cat((x, emb), dim=1)  # bs x 256 x n_p

        # translation branch
        t_x = F.relu(self.conv1_t(fusion))
        t_x = F.relu(self.conv2_t(t_x))
        t_global = F.adaptive_avg_pool1d(t_x, 1)    # bs x 1024 x 1
        t_feat = torch.cat((point_feat, t_global.repeat(1, 1, np)), dim=1)

        # rotation branch
        r_x = F.relu(self.conv1_r(fusion))
        r_x = F.relu(self.conv2_r(r_x))
        r_global = F.adaptive_avg_pool1d(r_x, 1)    # bs x 1024 x 1

        return t_feat, r_global


class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj, num_rot, k=16):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.num_rot = num_rot

        if self.num_rot == 12:
            self.rot_anchors = sample_rotations_12()
        elif self.num_rot == 24:
            self.rot_anchors = sample_rotations_24()
        elif self.num_rot == 60:
            self.rot_anchors = sample_rotations_60()
        else:
            raise NotImplementedError('num of rotation anchors must be 12, 24, or 60')

        self.cnn = ModifiedResnet()
        self.pointnet = ModifiedDGCNN(k)

        self.conv1_t = torch.nn.Conv1d(1152, 512, 1)
        self.conv2_t = torch.nn.Conv1d(512, 256, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1)

        self.conv1_r = torch.nn.Conv1d(1024, 512, 1)
        self.conv2_r = torch.nn.Conv1d(512, 256, 1)
        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv4_r = torch.nn.Conv1d(128, num_obj*num_rot*4, 1)

        self.conv1_c = torch.nn.Conv1d(1024, 512, 1)
        self.conv2_c = torch.nn.Conv1d(512, 256, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)
        self.conv4_c = torch.nn.Conv1d(128, num_obj*num_rot*1, 1)

    def forward(self, img, x, choose, obj):
        """ Only support batch size of 1

        Args:
            img: bs x 3 x H x W
            x: bs x n_p x 3
            choose: bs x n_p
            obj: bs x 1

        Returns:
            out_tx: 1 x n_p x 3
            out_rx: 1 x num_rot x 4
            out_cx: 1 x num_rot

        """
        # PSPNet: color feature extraction
        out_img = self.cnn(img)
        # concatenate color embedding with points
        bs, di, _, _ = out_img.size()
        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        x = x.permute(0, 2, 1)
        # DGCNN: point feature extraction
        t_feat, r_global = self.pointnet(x, emb)

        # prediction
        tx = F.relu(self.conv1_t(t_feat))
        tx = F.relu(self.conv2_t(tx))
        tx = F.relu(self.conv3_t(tx))
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)

        rx = F.relu(self.conv1_r(r_global))
        rx = F.relu(self.conv2_r(rx))
        rx = F.relu(self.conv3_r(rx))
        rx = self.conv4_r(rx).view(bs, self.num_obj, self.num_rot, 4)

        cx = F.relu(self.conv1_c(r_global))
        cx = F.relu(self.conv2_c(cx))
        cx = F.relu(self.conv3_c(cx))
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, self.num_rot)

        # select prediction from corresponding object class
        b = 0

        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        out_cx = torch.index_select(cx[b], 0, obj[b])   # 1 x num_rot

        out_rx = torch.index_select(rx[b], 0, obj[b])   # 1 x num_rot x 4
        out_rx = F.normalize(out_rx, p=2, dim=2)    # 1 x num_rot x 4
        rot_anchors = torch.from_numpy(self.rot_anchors).float().cuda()
        rot_anchors = torch.unsqueeze(torch.unsqueeze(rot_anchors, dim=0), dim=3)     # 1 x num_rot x 4 x 1
        out_rx = torch.unsqueeze(out_rx, 2)     # 1 x num_rot x 1 x 4
        out_rx = torch.cat((out_rx[:, :, :, 0], -out_rx[:, :, :, 1], -out_rx[:, :, :, 2], -out_rx[:, :, :, 3], \
                            out_rx[:, :, :, 1],  out_rx[:, :, :, 0],  out_rx[:, :, :, 3], -out_rx[:, :, :, 2], \
                            out_rx[:, :, :, 2], -out_rx[:, :, :, 3],  out_rx[:, :, :, 0],  out_rx[:, :, :, 1], \
                            out_rx[:, :, :, 3],  out_rx[:, :, :, 2], -out_rx[:, :, :, 1],  out_rx[:, :, :, 0], \
                            ), dim=2).contiguous().view(1, self.num_rot, 4, 4)
        out_rx = torch.squeeze(torch.matmul(out_rx, rot_anchors), dim=3)

        return out_rx, out_tx, out_cx
