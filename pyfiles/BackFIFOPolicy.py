import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import datetime
import datetime 
import RandAugment
import FastAutoAugment
import customUtils
import Augmix
import numpy as np
import localManagerCPU as PinData
# import .profile_data as pfdata
import statistics
import multiprocessing
from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT
from coatnet_pytorch import CoAtNet
import globalManager as gc
import psutil
CPU_NUM = psutil.cpu_count(logical = False)

EFFICIENT_INPUT_SIZE = {
    'efficientnet-b0': 224,
    'efficientnet-b1': 240,
    'efficientnet-b2': 260,
    'efficientnet-b3': 300,
    'efficientnet-b4': 380,
    'efficientnet-b5': 456,
    'efficientnet-b6': 528,
    'efficientnet-b7': 600,
    'efficientnet-b8': 672,
    'efficientnet-l2': 800
}

VIT_MODEL={
    "vit-base": {
            "image_size": 224,
            "patch_size": 16,
            "num_classes": 1000,
            "dim": 768,
            "depth": 12,
            "heads": 16,
            "mlp_dim": 3072,
            "dropout": 0.1,
            "emb_dropout": 0.1
    },

        "vit-large": {
            "image_size": 224,
            "patch_size": 16,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 24,
            "heads": 16,
            "mlp_dim": 4096,
            "dropout": 0.1,
            "emb_dropout": 0.1
        },

        "vit-huge" : {
            "image_size": 224,
            "patch_size": 16,
            "num_classes": 1000,
            "dim": 1280,
            "depth": 32,
            "heads": 16,
            "mlp_dim": 5120,
            "dropout": 0.1,
            "emb_dropout": 0.1
        }
}


if __debug__:
    import logging  
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
    
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--testdata', action='store_true',
                    help='Testing with small dataset')
parser.add_argument('--tracefetch', action='store_true',
                    help='Trace fetch')
parser.add_argument('--traceaug', action='store_true',
                    help='Trace aug')
parser.add_argument('--dotrain', default=0, type=int, metavar='N',
                    help='Do train for N seconds')
parser.add_argument('--fulltrace', action='store_true',
                    help='Testing with small dataset')
parser.add_argument('--augs', default='default', type=str, metavar='Augmentations',
help='Augmentation to run')
parser.add_argument('-t','--threads', default=0, type=int, metavar='N',
help='Number of data loading worker threads')

best_acc1 = 0

def seed_worker(worker_id):
    worker_seed = parser.parse_args().seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # NOTE if you want to GPU deterministic do it!
        # cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')
    if args.gpu is not None:    
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        
    traindir = os.path.join(args.data, 'train')
    test_loader =  torch.utils.data.DataLoader(
            customUtils.RdonlyDataset(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        sampler=None, batch_size=args.batch_size, shuffle=False,
        num_workers=CPU_NUM-1, pin_memory=False)
    
    for (images, target) in test_loader:
        pass

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        mp.spawn(main_worker, nprocs=1, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    if args.seed is not None:
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            batch_size = int(args.batch_size / ngpus_per_node)
            gc.globalManager_init(workers, batch_size, args.threads, controllerClass=gc.FIFOglobalManager, rank=args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.arch.startswith('vit'):
        model = ViT(**VIT_MODEL[args.arch])
    elif args.arch.startswith('efficientnet'):
        model = EfficientNet.from_pretrained(args.arch)
    elif args.arch.startswith('coatnet'):
        model = CoAtNet[args.arch]()
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.arch.startswith('vit'):
        size = 224
    elif args.arch.startswith('efficientnet'):
        size = EFFICIENT_INPUT_SIZE[args.arch]
    else:
        size = 224
    resize = size + 32

    augmentations=[
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
    
    if args.augs == "default":
        pass
    elif args.augs == "none":
        augmentations=[]
    elif args.augs == "randaugment":
        print(f"Using Randaugment {args.augs}")
        augmentations.insert(0, RandAugment.augmentations.RandAugment(2, 9))
    elif args.augs == "autoaugment":
        print(f"Using AutoAugment {args.augs}")
        augmentations.insert(0, FastAutoAugment.AutoAugment(FastAutoAugment.fa_resnet50_rimagenet()))
    elif args.augs == "croprandaugment":
        print(f"Using CropRandaugment {args.augs}")
        augmentations.insert(1, RandAugment.augmentations.RandAugment(2, 9))
    elif args.augs == "cropautoaugment":
        print(f"Using CropAutoAugment {args.augs}")
        augmentations.insert(1, FastAutoAugment.AutoAugment(FastAutoAugment.fa_resnet50_rimagenet()))
    elif args.augs == "norandom":
        print(f"Using Noromd default {args.augs}")
        augmentations=[
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize
            ]
    elif args.augs == "randaugmentnorandom":
        print(f"Using Noromd randaug {args.augs}")
        augmentations=[
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize
            ]
        augmentations.insert(1, RandAugment.augmentations.RandAugment(2, 9))
    elif args.augs == "fastaugment":
        # augmentations.insert(0, )
        raise Exception(f"Not supported augments: {args.augs}")
    else:
        raise Exception(f"Not supported augments: {args.augs}")


    # For CPU core pinning for the local GPU of each CPU Socket
    cur_rank = args.gpu
    import numa
    NODE_MAX = numa.get_max_node()
    NUM_GPU = gc.NUM_GPU
    GPU_PER_NODE = NUM_GPU // (NODE_MAX+1)
    GPU_NODE = args.gpu // GPU_PER_NODE
    CPU_IDS = list(numa.node_to_cpus(GPU_NODE))
    CPU_IDS_PER_GPU = CPU_NUM // NUM_GPU


    start_cpu_id = CPU_IDS_PER_GPU * args.gpu + args.workers

    end_cpu_id = start_cpu_id + 1
    p = psutil.Process()
    cpu_ids = []
    
    if args.traceaug:
        if __debug__:
            log.info("Tracaug on")
        augmentations_composed= customUtils.FullTraceCompose(augmentations)
    else:
        if __debug__:
            log.info("Tracaug off")
        augmentations_composed= transforms.Compose(augmentations)
    
    if args.tracefetch:
        train_dataset = customUtils.FullTraceImageFolder(
            cur_rank,
            traindir,
            augmentations_composed
        )
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            augmentations_composed
        )
        
    if args.distributed:    
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    # FIXME hardcode for image
    template_shape = (args.batch_size, 3, size, size)
    if args.seed is None:
        if __debug__:
            log.info(f"No seed")
        train_loader = PinData.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), rank = cur_rank,
                num_workers=args.workers, pin_memory=False, persistent_workers=True, sampler=train_sampler, drop_last=True, worker_batch_size=args.threads)
    else:
        if __debug__:
            log.info(f"Seed {args.seed}")

        g = torch.Generator()
        g.manual_seed(args.seed)
        train_loader = PinData.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), rank = cur_rank,
                num_workers=args.workers, pin_memory=False, persistent_workers=True, sampler=train_sampler, drop_last=True, worker_init_fn=seed_worker, generator=g, worker_batch_size=args.threads)
    
    # test_loader =  torch.utils.data.DataLoader(
    #         datasets.ImageFolder(traindir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(size),
    #         transforms.ToTensor(),
    #     ])),
    #     sampler=train_sampler, batch_size=args.batch_size, shuffle=False,
    #     num_workers=(CPU_NUM//ngpus_per_node - 1), pin_memory=False)
    
    # val_loader = torch.utils.data.DataLoader(
    #             datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(size),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=False)

    val_loader=None
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    # for (images, target) in test_loader:
    #     pass

    for i in range(start_cpu_id, end_cpu_id):
        cpu_ids.append(i) 


    p.cpu_affinity(cpu_ids)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = 0 # validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            #save_checkpoint({
             #   'epoch': epoch + 1,
              #  'arch': args.arch,
               # 'state_dict': model.state_dict(),
                #'best_acc1': best_acc1,
                #'optimizer' : optimizer.state_dict(),
            #}, is_best)
            pass


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.6f')
    data_time = AverageMeter('Data', ':6.6f')
    throughput = AverageMeter('Throughput', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, throughput],
        prefix=f"GPU: {args.gpu} Epoch: [{epoch}]")
        # [batch_time, data_time, throughput, losses, top1, top5],

    # switch to train mode
    model.train()
    
    data_arr = [] 
    batch_arr = []
    dtimes = []
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data = torch.cuda.Event(enable_timing=True)
        data.record()
        dtime = datetime.datetime.now()
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.to(args.gpu, non_blocking=True, dtype=torch.long)
        
        with torch.cuda.amp.autocast():
            # compute output
            
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            # train_loader._worker_control()
            loss.backward()
            optimizer.step()
        
                
        if __debug__:
            log.debug(f"GPU {args.gpu} Iteration{i} done")
        print(f"GPU {args.gpu} Iteration{i} done")
        # measure elapsed time
        # torch.cuda.synchronize() 
        batch = torch.cuda.Event(enable_timing=True)
        batch.record()
        data_arr.append((end,data))
        batch_arr.append((end,batch))
        dtimes.append(dtime)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        
    if __debug__:
        log.debug(f"GPU {args.gpu} Arrange time for iterations output")
    
    for i in range(len(data_arr)):
        data_arr[i][0].synchronize()
        # data_arr[i][1].synchronize()
        datatime = data_arr[i][0].elapsed_time(data_arr[i][1])/1000
        data_time.update(datatime)
        # batch_arr[i][0].synchronize()
        batch_arr[i][1].synchronize()
        itertime = batch_arr[i][0].elapsed_time(batch_arr[i][1])/1000
        batch_time.update(itertime)

        throughput.update(args.batch_size/itertime)
        progress.display(i, dtimes[i])

    if __debug__:
        log.debug(f"GPU {args.gpu} Arrnage time for iterations done")
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (images, target) in enumerate(val_loader):
            dtime = datetime.datetime.now()
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if i % args.print_freq == 0:
                progress.display(i, dtime)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, dtime):
        entries = [f'{dtime} ' + self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
