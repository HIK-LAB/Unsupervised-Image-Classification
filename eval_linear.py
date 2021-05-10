import argparse
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from UIC.util import AverageMeter, create_logger
from UIC import models

parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                 of frozen convolutional layers of an AlexNet.""")
parser.add_argument('--data', type=str, help='path to dataset')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--epochs', type=int, default=32, help='number of total epochs to run')
parser.add_argument('--exp', type=str, default='', help='exp folder')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=3000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--save_name', default='LinearEval')
parser.add_argument('--seed', type=int, default=31, help='random seed')
parser.add_argument('--step_size', type=int, default=10, help='step_size')
parser.add_argument('--tencrops', action='store_true',
                    help='validation accuracy averaged over 10 crops')
parser.add_argument('--verbose', action='store_true', help='chatty')
parser.add_argument('--weight_decay', '--wd', default=-4, type=float,
                    help='weight decay pow (default: -4)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

def main():
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
    best_prec1 = 0

    # network defined
    checkpoint = torch.load(args.model)
    model = models.__dict__[checkpoint['arch']](out=args.nmb_cluster, linear_eval=True, extra_mlp=True)

    # freeze the features layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = True

    # load model
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    cudnn.benchmark = True

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # train & val dataloader
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transformations_train = [transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize]
    if args.tencrops:
        transformations_val = [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
        ]
    else:
        transformations_val = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]
    train_dataset = datasets.ImageFolder(
        traindir,
        transform=transforms.Compose(transformations_train)
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transform=transforms.Compose(transformations_val)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(args.batch_size//2),
        shuffle=False,
        num_workers=4
    )

    # optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.module.linear.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=10**args.weight_decay if args.weight_decay != 0 else 0
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size, 
        gamma=0.1, 
        last_epoch=-1
    )

    for epoch in range(args.epochs):
        end = time.time()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec5, loss = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            filename = 'model_best.pth.tar'
        else:
            filename = 'checkpoint_latest.pth.tar'
        torch.save({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'prec5': prec5,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(args.exp, filename))
        scheduler.step()

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # freeze also batch norm layers
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        input_cu, target_cu = input.cuda(), target.cuda()
        output = model(input_cu)
        loss = criterion(output, target_cu)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_cu, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and i % 100 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'lr {3}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                        .format(epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], \
                          batch_time=batch_time, data_time=data_time, loss=losses, \
                          top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    softmax = nn.Softmax(dim=1).cuda()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.tencrops:
                bs, ncrops, c, h, w = input.size()
                input = input.view(-1, c, h, w).contiguous()
            input_cu, target_cu = input.cuda(), target.cuda()
            
            output = model(input_cu)
            if args.tencrops:
                output_central = output.view(bs, ncrops, -1)[: , int(ncrops / 2 - 1), :]
                output = softmax(output)
                output = torch.squeeze(output.view(bs, ncrops, -1).mean(1))
            else:
                output_central = output

            prec1, prec5 = accuracy(output.data, target_cu, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            loss = criterion(output_central, target_cu)
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info('Validation: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                .format(i, len(val_loader), batch_time=batch_time,
                 loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

if __name__ == '__main__':
    main()
