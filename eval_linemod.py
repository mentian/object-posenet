import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset
from lib.network import PoseNet
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.transformations import quaternion_matrix
from lib.knn.__init__ import KNearestNeighbor


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
parser.add_argument('--model', type=str, default='trained_models/linemod/pose_model_40_0.009487.pth',  help='Evaluation model')
parser.add_argument('--dataset_root', type=str, default='datasets/linemod/Linemod_preprocessed', help='dataset root dir')
opt = parser.parse_args()

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
num_rotations = 60
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'results/eval_linemod'
if not os.path.exists(output_result_dir):
    os.makedirs(output_result_dir)
knn = KNearestNeighbor(1)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
estimator = PoseNet(num_points=num_points, num_obj=num_objects, num_rot=num_rotations)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

test_dataset = PoseDataset('eval', num_points, False, opt.dataset_root, 0.0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
sym_list = test_dataset.get_sym_list()
rot_anchors = torch.from_numpy(estimator.rot_anchors).float().cuda()
diameter = test_dataset.get_diameter()

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

error_data = 0
for i, data in enumerate(test_dataloader, 0):
    try:
        points, choose, img, target_t, target_r, model_points, idx, gt_t = data
    except:
        error_data += 1
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points, choose, img, target_t, target_r, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(target_t).cuda(), \
                                                                 Variable(target_r).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
    pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
    pred_t, pred_mask = ransac_voting_layer(points, pred_t)
    pred_t = pred_t.cpu().data.numpy()
    how_min, which_min = torch.min(pred_c, 1)
    pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
    pred_r = quaternion_matrix(pred_r)[:3, :3]
    model_points = model_points[0].cpu().detach().numpy()
    pred = np.dot(model_points, pred_r.T) + pred_t
    target = target_r[0].cpu().detach().numpy() + gt_t.cpu().data.numpy()[0]

    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    if dis < 0.1 * diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1

accuracy = 0.0
for i in range(num_objects):
    accuracy += float(success_count[i]) / num_count[i]
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
print('Accuracy: {0}'.format(accuracy / num_objects))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.write('Accuracy: {0}\n'.format(accuracy / num_objects))
fw.write('{0} corrupted data'.format(error_data))
fw.close()
