import argparse
import copy as cp
import numpy as np
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.nn import functional as F
from torchvision import transforms as T
from UIC import models
from UIC.dataset import DatasetGivenLabels
from UIC.sampler import UnifLabelSampler
from UIC.transform import color_distortion
from UIC.util import AverageMeter, create_logger

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='resnet50',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=3000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--save_name', default='UnsupervisedImageClassification')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--suffix', default='JPEG')
parser.add_argument('--verbose', action='store_true', help='chatty')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--workers', default=12, type=int,
                    help='number of data loading workers (default: 4)')
        

def main():
    # log-file setting
    global args
    args = parser.parse_args()
    log_file_name = args.save_name + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log'
    global logger
    logger = create_logger(os.path.join(args.exp, log_file_name))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment will be stored in %s\n" % args.exp)
    logger.info("")

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        logger.info('Architecture: {}'.format(args.arch))
    # extra mlp head & random gaussian bluring augmentation
    model = models.__dict__[args.arch](out=args.nmb_cluster, extra_mlp=True, random_gblur=True)
    model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, \
        eta_min=0, last_epoch=-1)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    end = time.time()
    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip(),
           color_distortion(s=1.0),
           transforms.ToTensor(),
           normalize]
    tra_test = [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize]

    # load the data
    dataset = DatasetGivenLabels(args.data, transform=transforms.Compose(tra_test), suffix=args.suffix)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    # initialize labels
    logger.info('initializing labels')
    labels = compute_labels(dataloader, model, args.nmb_cluster, len(dataset),\
        init_via_forward=(args.resume and os.path.isfile(args.resume)))
    history_labels = cp.copy(labels)
    nmi = normalized_mutual_info_score(labels, dataset.gt)
    logger.info('NMI against gt assignment: {0:.3f}'.format(nmi))

    if args.verbose: logger.info('Load dataset: {0:.2f} s'.format(time.time() - end))
    for epoch in range(start_epoch, args.epochs):
        end = time.time()

        # re-arrange labels to avoid empty classes
        images_lists = [[] for _ in range(args.nmb_cluster)]
        for i, label_item in enumerate(labels):
            images_lists[int(label_item)].append(i)
        empty_list = [i for i, item in enumerate(images_lists) if len(item) == 0]
        if len(empty_list) > 0:
            logger.info('empty class number: {}'.format(len(empty_list)))
        for empty_idx in empty_list:
            max_idx = np.argmax([len(item) for item in images_lists])
            max_item = images_lists[max_idx]
            np.random.shuffle(max_item)
            images_lists[empty_idx] = max_item[len(max_item)//2:]
            images_lists[max_idx] = max_item[:len(max_item)//2]
        labels = np.zeros(len(labels))
        for i, item in enumerate(images_lists):
            for j in item:
                labels[j] = i

        # balanced sampling dataloader
        train_dataset = DatasetGivenLabels(args.data, labels, \
            transform=transforms.Compose(tra), suffix=args.suffix)
        sampler = UnifLabelSampler(int(len(train_dataset)), images_lists)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # train network and update pseudo labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch, labels, label_update=True)

        # logger.info log
        if args.verbose:
            logger.info('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'ConvNet loss: {2:.3f}'
                  .format(epoch, time.time() - end, loss))
            try:
                nmi = normalized_mutual_info_score(labels, history_labels)
                logger.info('NMI against previous assignment: {0:.3f}'.format(nmi))
                nmi = normalized_mutual_info_score(labels, dataset.gt)
                logger.info('NMI against gt assignment: {0:.3f}'.format(nmi))
            except:
                pass
            logger.info('####################### \n')

        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint_latest.pth.tar'))
        history_labels = cp.copy(labels)
        lr_scheduler.step()

def train(loader, model, crit, opt, epoch, labels, label_update=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets, index) in enumerate(loader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)

        # labels updated
        if label_update:
            predict = outputs.argmax(dim=1).data.cpu().numpy()
            for index_item, pred in zip(index, predict):
                labels[index_item] = int(pred)

        loss = crit(outputs, targets)

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.verbose and (i % 100) == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'LR: {3}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), opt.param_groups[0]['lr'], batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg

def compute_labels(dataloader, model, class_num, data_len, init_via_forward=False):
    '''Pre-generate pseudo labels via network forward or uniformly assignment'''
    if args.verbose:
        logger.info('Compute labels')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if init_via_forward:
        model.eval()
        label_list = []
        for i, inputs in enumerate(dataloader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            with torch.no_grad():
                output = model(inputs)
            batch_time.update(time.time() - end)

            label = output.argmax(dim=1).cpu()
            if i == 0:
                label_list = label
            else:
                label_list = torch.cat([label_list, label], dim=0)
            if args.verbose and (i % 100) == 0:
                logger.info('{0}/{1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      .format(i, len(dataloader), batch_time=batch_time, data_time=data_time))
            end = time.time()
        label_list = label_list.numpy()
        model.train()
    else:
        label_list = np.array([int(np.random.uniform(0, class_num)) for _ in range(data_len)])

    return label_list

if __name__ == '__main__':
    main()