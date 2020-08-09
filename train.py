import os
import argparse
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
import tensorflow as tf
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet
from lib.loss import Loss
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.transformations import quaternion_matrix
from lib.knn.__init__ import KNearestNeighbor
from lib.utils import setup_logger


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='linemod', help='ycb or linemod')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
parser.add_argument('--num_rot', type=int, default=60, help='number of rotation anchors')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--noise_trans', default=0.01, help='random noise added to translation')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
parser.add_argument('--nepoch', type=int, default=50, help='max number of epochs to train')
opt = parser.parse_args()


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.dataset == 'ycb':
        opt.dataset_root = 'datasets/ycb/YCB_Video_Dataset'
        opt.num_objects = 21
        opt.num_points = 1000
        opt.result_dir = 'results/ycb'
        opt.repeat_epoch = 1
    elif opt.dataset == 'linemod':
        opt.dataset_root = 'datasets/linemod/Linemod_preprocessed'
        opt.num_objects = 13
        opt.num_points = 500
        opt.result_dir = 'results/linemod'
        opt.repeat_epoch = 10
    else:
        print('unknown dataset')
        return
    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans)
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans)
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    opt.diameters = dataset.get_diameter()
    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<')
    print('length of the training set: {0}'.format(len(dataset)))
    print('length of the testing set: {0}'.format(len(test_dataset)))
    print('number of sample points on mesh: {0}'.format(opt.num_points_mesh))
    print('symmetrical object list: {0}'.format(opt.sym_list))

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    tb_writer = tf.summary.FileWriter(opt.result_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # network
    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects, num_rot=opt.num_rot)
    estimator.cuda()
    # loss
    criterion = Loss(opt.sym_list, estimator.rot_anchors)
    knn = KNearestNeighbor(1)
    # learning rate decay
    best_test = np.Inf
    opt.first_decay_start = False
    opt.second_decay_start = False
    # if resume training
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load(opt.resume_posenet))
        model_name_parsing = (opt.resume_posenet.split('.')[0]).split('_')
        best_test = float(model_name_parsing[-1])
        opt.start_epoch = int(model_name_parsing[-2]) + 1
        if best_test < 0.016 and not opt.first_decay_start:
            opt.first_decay_start = True
            opt.lr *= 0.6
        if best_test < 0.013 and not opt.second_decay_start:
            opt.second_decay_start = True
            opt.lr *= 0.5
    # optimizer
    optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)
    global_step = (len(dataset) // opt.batch_size) * opt.repeat_epoch * (opt.start_epoch - 1)
    # train
    st_time = time.time()
    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%02d' % epoch, os.path.join(opt.result_dir, 'epoch_%02d_train_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_loss_avg = 0.0
        train_loss_r_avg = 0.0
        train_loss_t_avg = 0.0
        train_loss_reg_avg = 0.0
        estimator.train()
        optimizer.zero_grad()
        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target_t, target_r, model_points, idx, gt_t = data
                obj_diameter = opt.diameters[idx]
                points, choose, img, target_t, target_r, model_points, idx = Variable(points).cuda(), \
                                                                             Variable(choose).cuda(), \
                                                                             Variable(img).cuda(), \
                                                                             Variable(target_t).cuda(), \
                                                                             Variable(target_r).cuda(), \
                                                                             Variable(model_points).cuda(), \
                                                                             Variable(idx).cuda()
                pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
                loss, loss_r, loss_t, loss_reg = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter)
                loss.backward()
                train_loss_avg += loss.item()
                train_loss_r_avg += loss_r.item()
                train_loss_t_avg += loss_t.item()
                train_loss_reg_avg += loss_reg.item()
                train_count += 1
                if train_count % opt.batch_size == 0:
                    global_step += 1
                    lr = opt.lr
                    optimizer.step()
                    optimizer.zero_grad()
                    # write results to tensorboard
                    summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate', simple_value=lr),
                                                tf.Summary.Value(tag='loss', simple_value=train_loss_avg / opt.batch_size),
                                                tf.Summary.Value(tag='loss_r', simple_value=train_loss_r_avg / opt.batch_size),
                                                tf.Summary.Value(tag='loss_t', simple_value=train_loss_t_avg / opt.batch_size),
                                                tf.Summary.Value(tag='loss_reg', simple_value=train_loss_reg_avg / opt.batch_size)])
                    tb_writer.add_summary(summary, global_step)
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_loss:{4:f}'.format(time.strftime("%Hh %Mm %Ss", 
                        time.gmtime(time.time()-st_time)), epoch, int(train_count/opt.batch_size), train_count, train_loss_avg/opt.batch_size))
                    train_loss_avg = 0.0
                    train_loss_r_avg = 0.0
                    train_loss_t_avg = 0.0
                    train_loss_reg_avg = 0.0

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%02d_test' % epoch, os.path.join(opt.result_dir, 'epoch_%02d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        save_model = False
        estimator.eval()
        success_count = [0 for i in range(opt.num_objects)]
        num_count = [0 for i in range(opt.num_objects)]

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target_t, target_r, model_points, idx, gt_t = data
            obj_diameter = opt.diameters[idx]
            points, choose, img, target_t, target_r, model_points, idx = Variable(points).cuda(), \
                                                                         Variable(choose).cuda(), \
                                                                         Variable(img).cuda(), \
                                                                         Variable(target_t).cuda(), \
                                                                         Variable(target_r).cuda(), \
                                                                         Variable(model_points).cuda(), \
                                                                         Variable(idx).cuda()
            pred_r, pred_t, pred_c = estimator(img, points, choose, idx)
            loss, _, _, _ = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter)
            test_count += 1
            # evalaution
            how_min, which_min = torch.min(pred_c, 1)
            pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
            pred_r = quaternion_matrix(pred_r)[:3, :3]
            pred_t, pred_mask = ransac_voting_layer(points, pred_t)
            pred_t = pred_t.cpu().data.numpy()
            model_points = model_points[0].cpu().detach().numpy()
            pred = np.dot(model_points, pred_r.T) + pred_t
            target = target_r[0].cpu().detach().numpy() + gt_t[0].cpu().data.numpy()
            if idx[0].item() in opt.sym_list:
                pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
                target = torch.index_select(target, 1, inds.view(-1) - 1)
                dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
            else:
                dis = np.mean(np.linalg.norm(pred - target, axis=1))
            logger.info('Test time {0} Test Frame No.{1} loss:{2:f} confidence:{3:f} distance:{4:f}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss, how_min[0].item(), dis))
            if dis < 0.1 * opt.diameters[idx[0].item()]:
                success_count[idx[0].item()] += 1
            num_count[idx[0].item()] += 1
            test_dis += dis
        # compute accuracy
        accuracy = 0.0
        for i in range(opt.num_objects):
            accuracy += float(success_count[i]) / num_count[i]
            logger.info('Object {0} success rate: {1}'.format(test_dataset.objlist[i], float(success_count[i]) / num_count[i]))
        accuracy = accuracy / opt.num_objects
        test_dis = test_dis / test_count
        # log results
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2:f}, Accuracy: {3:f}'.format(time.strftime("%Hh %Mm %Ss",
            time.gmtime(time.time() - st_time)), epoch, test_dis, accuracy))
        # tensorboard
        summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                                    tf.Summary.Value(tag='test_dis', simple_value=test_dis)])
        tb_writer.add_summary(summary, global_step)
        # save model
        if test_dis < best_test:
            best_test = test_dis
        torch.save(estimator.state_dict(), '{0}/pose_model_{1:02d}_{2:06f}.pth'.format(opt.result_dir, epoch, best_test))
        # adjust learning rate if necessary
        if best_test < 0.016 and not opt.first_decay_start:
            opt.first_decay_start = True
            opt.lr *= 0.6
            optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)
        if best_test < 0.013 and not opt.second_decay_start:
            opt.second_decay_start = True
            opt.lr *= 0.5
            optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)

        print('>>>>>>>>----------epoch {0} test finish---------<<<<<<<<'.format(epoch))


if __name__ == '__main__':
    main()
