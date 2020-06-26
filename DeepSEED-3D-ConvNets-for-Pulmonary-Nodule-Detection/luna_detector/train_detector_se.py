import argparse
import os
import time
import numpy as np
from data_loader import collate
from lidc_dataset import LIDCDataset
from importlib import import_module
import shutil
from utils import *
import sys
from data_loader import LungNodule3Ddetector

sys.path.append('../')
from split_combine import SplitComb
import pdb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
import torch.autograd
from config_training import config as config_training

from layers_se import acc

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='res18_se',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='10', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='test_results/se_focal_fold6/', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=1, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--train_len', default=400, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--val_len', default=50, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--verbose', default=False, type=bool, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--random', default=False, type=bool, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--mode', default='ours', type=str, metavar='N',
                    help='either our lidc version or their luna version')
parser.add_argument('--with_augmented', default=0, type=int, metavar='N',
                    help='either include augmented data in train set or not')
parser.add_argument('--dataset_split', default=0, type=int, metavar='N',
                    help='split mode')


def main():
    global args
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    print('creating model')
    config, net, loss, get_pbb = model.get_model()
    print('created model')
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    config['verbose'] = args.verbose

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load('test_results/{}'.format(args.resume))
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)

    if args.dataset_split == 0:
        split_files_postfix = ''
    else:
        split_files_postfix = '_{}'.format(args.dataset_split)


    luna_train = np.load('./luna_train{}.npy'.format(split_files_postfix))
    luna_test = np.load('./luna_test{}.npy'.format(split_files_postfix))


    if args.mode != 'ours':
        datadir = os.path.join('/content/drive/My Drive/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection',
                               config_training['preprocess_result_path'])
        print('len lun train', len(luna_train))
        with_augmented = args.with_augmented == 1
        train_dataset = LungNodule3Ddetector(datadir, luna_train, config, phase='train', with_augmented=with_augmented)
    else:
        datadir = '/content/drive/My Drive/dsb2018_topcoders/data'
        train_dataset = LIDCDataset(datadir, config, 0, args.train_len, load=True, random=args.random)

    print(args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)


    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)

    def get_lr(epoch):
        if epoch <= args.epochs * 0.2:
            lr = args.lr
        elif epoch <= args.epochs * 0.4:
            lr = 0.1 * args.lr
        elif epoch <= args.epochs * 0.6:
            lr = 0.05 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    best_loss = 100
    for epoch in range(start_epoch, args.epochs + 1):
        train(train_loader, net, loss, epoch, optimizer, get_lr, save_dir)
        print("finsihed epoch {}".format(epoch))
        # vali_loss = validate(val_loader, net, loss)

        # if best_loss > vali_loss:
        #     best_loss = vali_loss
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        if epoch % 3 == 2:
            torch.save({
                'epoch': epoch + 1,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir, 'detector_%03d.ckpt' % epoch))
            print("save model on epoch %d" % epoch)


def resolve_dataset_split(dataset_split):
    if dataset_split == 0:
        return np.load('./luna_train.npy')

def train(data_loader, net, loss, epoch, optimizer, get_lr, save_dir):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = torch.autograd.Variable(data.cuda(non_blocking=True))
        target = torch.autograd.Variable(target.cuda(non_blocking=True))
        coord = torch.autograd.Variable(coord.cuda(non_blocking=True))

        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.FloatTensor)
        coord = coord.type(torch.cuda.FloatTensor)

        # print('data shape: {}'.format(data.shape))
        # print('coord shape: {}'.format(coord.shape))
        output = net(data, coord)

        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

        # print("finished iteration {} with loss {}.".format(i, loss_output[0]))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))


if __name__ == '__main__':
    main()
