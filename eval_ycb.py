import os
import time
import argparse
from PIL import Image
import numpy as np
import numpy.ma as ma
import scipy.io as scio
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.network import PoseNet
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
parser.add_argument('--model', type=str, default='trained_models/ycb/pose_model_42_0.010920.pth',  help='Evaluation model')
parser.add_argument('--dataset_root', type=str, default='datasets/ycb/YCB_Video_Dataset', help='dataset root dir')
opt = parser.parse_args()

norm = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_rotations = 60
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
output_result_dir = 'YCB_Video_toolbox/my_result'
if not os.path.exists(output_result_dir):
    os.mkdir(output_result_dir)


def get_bbox(posecnn_rois):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
estimator = PoseNet(num_points=num_points, num_obj=num_obj, num_rot=num_rotations)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()
rot_anchors = torch.from_numpy(estimator.rot_anchors).float().cuda()

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

t_start = time.time()
obj_count = 0
pe_time = 0
for now in range(0, 2949):
    img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    label = np.array(posecnn_meta['labels'])
    posecnn_rois = np.array(posecnn_meta['rois'])
    lst = posecnn_rois[:, 1:2].flatten()

    my_result = []
    for idx in range(len(lst)):
        itemid = lst[idx]
        try:
            t1 = time.time()
            obj_count += 1
            # crop object from image
            rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)
            img_masked = np.array(img)[:, :, :3]
            img_masked = img_masked[rmin:rmax, cmin:cmax, :]
            # obtain point cloud
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                assert len(choose) > 1
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])
            # point cloud
            pt2 = depth_masked / cam_scale
            pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            # estimation
            cloud = Variable(torch.from_numpy(cloud.astype(np.float32))).cuda()
            choose = Variable(torch.LongTensor(choose.astype(np.int32))).cuda()
            img_masked = Variable(norm(img_masked)).cuda()
            index = Variable(torch.LongTensor([itemid-1])).cuda()
            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
            pred_r, pred_t, pred_c = estimator(img_masked, cloud, choose, index)
            # evaluation
            try:
                pred_t, pred_mask = ransac_voting_layer(cloud, pred_t)
            except RuntimeError:
                print('RANSAC voting fails {0} at No.{1} keyframe'.format(itemid, now))
                my_result.append([0.0 for i in range(7)])
                continue
            my_t = pred_t.cpu().data.numpy()
            how_min, which_min = torch.min(pred_c, 1)
            my_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            my_result.append(my_pred.tolist())

            t2 = time.time()
            pe_time += (t2 - t1)

        except AssertionError:
            print('PoseNet Detector Lost {0} at No.{1} keyframe'.format(itemid, now))
            my_result.append([0.0 for i in range(7)])

    scio.savemat('{0}/{1}.mat'.format(output_result_dir, '%04d' % now), {'poses': my_result})
    print("Finish No.{0} keyframe".format(now))

t_end = time.time()
print('Average running time per frame: {}'.format((t_end-t_start)/2949))
print('Average PE time per object: {}'.format(pe_time/obj_count))
