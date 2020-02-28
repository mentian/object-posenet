import random
import numpy as np
import numpy.ma as ma
import scipy.io as scio
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data


class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans):
        self.objlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        if mode == 'train':
            path = 'datasets/ycb/dataset_config/train_data_list.txt'
        elif mode == 'test':
            path = 'datasets/ycb/dataset_config/test_data_list.txt'
        self.root = root
        self.list = []
        self.real = []
        self.syn = []
        # read data list
        input_file = open(path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
                # self.list.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()
        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)
        # read object model
        class_file = open('datasets/ycb/dataset_config/classes.txt')
        class_id = 1
        self.cld = {}
        self.diameter = []
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            input_file = open('{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            obj_cld = np.array(self.cld[class_id])
            self.cld[class_id] = obj_cld
            input_file.close()
            # compute object diameter
            obj_center = (np.amin(obj_cld, axis=0) + np.amax(obj_cld, axis=0)) / 2.0
            obj_cld = obj_cld - obj_center
            obj_diameter = np.amax(np.linalg.norm(obj_cld, axis=1)) * 2
            self.diameter.append(obj_diameter)
            class_id += 1
        class_file.close()

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])

        self.num_pt = num_pt
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh = 500
        self.front_num = 2

        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        print('Number of data: {0}'.format(len(self.list)))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        # camera intrinsic parameters
        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))
        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                    continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 5000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)
        for idx in random.sample(range(len(obj)), len(obj)):
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > 1000:
                break

        # crop object
        if self.add_noise:
            img = self.trancolor(img)
        try:
            rmin, rmax, cmin, cmax = get_bbox(mask_label)
        except IndexError:
            print('Error data: {0}-color.png'.format(self.list[index]))
            exit()
        img = np.array(img)[rmin:rmax, cmin:cmax, :3]
        # change background if data from syntentic image
        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
            back = back[rmin:rmax, cmin:cmax, :]
            # synthetic image black background, therefore directly add img
            img_masked = back * mask_back[rmin:rmax, cmin:cmax, None] + img
        else:
            img_masked = img
        # add foreground occlusion
        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax, None] + front[rmin:rmax, cmin:cmax, :] * ~(mask_front[rmin:rmax, cmin:cmax, None])
        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)
        img_masked = np.clip(img_masked, 0, 255).astype(np.uint8)

        # target
        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = meta['poses'][:, :, idx][:, 3:4].flatten()

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        # point cloud
        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            # shift
            add_t = np.random.uniform(-self.noise_trans, self.noise_trans, (1, 3))
            target_t = target_t + add_t
            # jittering
            add_t = add_t + np.clip(0.001*np.random.randn(cloud.shape[0], 3), -0.005, 0.005)
            cloud = np.add(cloud, add_t)
        # position target
        gt_t = target_t
        target_t = target_t - cloud
        target_t = target_t / np.linalg.norm(target_t, axis=1)[:, None]

        # rotation target
        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)
        target_r = np.dot(model_points, target_r.T)

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.transform(img_masked), \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([int(obj[idx]) - 1]), \
               torch.from_numpy(gt_t.astype(np.float32))

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh

    def get_diameter(self):
        return self.diameter


def get_bbox(label):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
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
