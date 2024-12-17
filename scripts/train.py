from __future__ import print_function, division
import os, pickle, sys
import numpy as np
import os.path as path
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision

import sys
sys.path.append('../') # path to SSE-SAM folder
from utils.eval_funcs import print_accuracy
from utils.datasets_LT import CIFAR100LT, CIFAR10LT, get_img_num_per_cls, gen_imbalanced_data
from utils.network_arch_resnet import ResnetEncoder
from utils.trainval import train_model
import warnings # ignore warnings
warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)
from ssesam import SAM, ImbSAM, SSESAM
from utils.loss_funcs import LDAMLoss, VSLoss

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9826)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'LDAM', 'LA', 'VS'])

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('-bz', '--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imb_factor', type=float, default=0.01)

    # optimizer
    parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'sam', 'imbsam', 'ssesam'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.001)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--head_rho', type=float, default=0.05)
    parser.add_argument('--tail_rho', type=float, default=0.05)
    parser.add_argument('--eta', type=int, default=20)
    parser.add_argument('--reweight', type=float, default=1.)
    parser.add_argument('--gamma', type=float, default=0)

    args = parser.parse_args()
    
    return args

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set device, which gpu to use.
    device ='cpu'
    if torch.cuda.is_available(): 
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.device_count()
        torch.cuda.empty_cache()
        
    if args.dataset == 'cifar100':
        no_of_classes = 100
        dataloaders, train_samples_per_cls, train_few_classes, train_labelList = prepare_data_cifar100(args, no_of_classes)
    elif args.dataset == 'cifar10':
        no_of_classes = 10
        dataloaders, train_samples_per_cls, train_few_classes, train_labelList = prepare_data_cifar10(args, no_of_classes)

    print('{} train dataset have {} samples for the head class and {} samples for tail class.'.format(args.dataset, max(train_samples_per_cls), min(train_samples_per_cls)))
    
    # get the execute path
    curr_working_dir = os.getcwd()
    
    model_name = args.loss

    opt_name = args.opt
    if opt_name == 'sam':
        opt_name = os.path.join(opt_name, 'rho{}'.format(args.rho))
    if opt_name == 'imbsam' or opt_name == 'ssesam':
        opt_name = os.path.join(opt_name, 'eta{}'.format(args.eta))
    if opt_name == 'ssesam':
        opt_name = os.path.join(opt_name, 'head_rho{}'.format(args.head_rho))
        opt_name = os.path.join(opt_name, 'tail_rho{}'.format(args.tail_rho))
        opt_name = os.path.join(opt_name, 'gamma{}'.format(args.ysam_cut))
        
    save_dir = path.join(curr_working_dir, 'work_dir', args.dataset,
                         model_name,
                         opt_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    model = ResnetEncoder(34, False, embDimension=no_of_classes, poolSize=4).to(device)
    
    train(args, device, model_name, model, save_dir, dataloaders, train_few_classes, train_labelList, train_samples_per_cls)
    test(args, device, model_name, save_dir, dataloaders, train_labelList, model)

def test(args, device, model_name, save_dir, dataloaders, train_labelList, model):
    path_to_clsnet = os.path.join(save_dir, '{}_best_{}.pth'.format(model_name, args.imb_factor))
    model.load_state_dict(torch.load(path_to_clsnet, map_location=device))
    model = model.to(device)

    print('Testing....'.format(model_name))
    print_accuracy(model, dataloaders, train_labelList, epoch = args.epochs, dataset = args.dataset, device = device, save_dir=save_dir)

def train(args, device, model_name, model, save_dir, dataloaders, train_few_classes, train_labelList, train_samples_per_cls):
    
    if model_name=='CE':
        loss_func = nn.CrossEntropyLoss(reduction='none').to(device)
    elif model_name=='LDAM':
        loss_func = LDAMLoss(cls_num_list=torch.tensor(train_samples_per_cls, device=device))
    elif model_name=='LA':
        loss_func = VSLoss(cls_num_list=train_samples_per_cls, device=device, gamma=0, tau=0.75)
    elif model_name=='VS':
        loss_func = VSLoss(cls_num_list=train_samples_per_cls, device=device, gamma=0.05, tau=0.75)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    scheduler = [lr_scheduler]
    
    if args.opt == 'sam':
        optimizer = SAM(optimizer=optimizer, model=model, rho=args.rho)
    elif args.opt == 'imbsam':
        optimizer = ImbSAM(optimizer=optimizer, model=model, rho=args.rho)
    elif args.opt == 'ssesam':
        optimizer = SSESAM(optimizer=optimizer, model=model, head_rho=args.head_rho, tail_rho=args.tail_rho, gamma=args.gamma, total_epochs=args.epochs)

    train_model(dataloaders, model, loss_func, optimizer, scheduler, 
                num_epochs=args.epochs, model_name= model_name, work_dir=save_dir, 
                device=device, print_each=args.print_freq,
                optim_mode=args.opt, tail_classes=train_few_classes, train_labelList=train_labelList, 
                reweight=args.reweight, imb_factor=args.imb_factor, dataset=args.dataset)

    return model

def prepare_data_cifar100(args, no_of_classes, begin_class=-1, class_step=0):
    path_to_DB = './datasets'
    if not os.path.exists(path_to_DB): os.makedirs(path_to_DB)
    _ = torchvision.datasets.CIFAR100(root=path_to_DB, train=True, download=True)
    indices = range(no_of_classes)
    if begin_class != -1:
        if class_step == 0:
            indices = [begin_class]
        else:
            indices = range(begin_class, no_of_classes, class_step)

    path_to_DB = path.join(path_to_DB, 'cifar-100-python')

    datasets = {}
    dataloaders = {}

    setname = 'meta'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        labelnames = pickle.load(obj, encoding='bytes')
        labelnames = labelnames[b'fine_label_names']
    for i in range(len(labelnames)):
        labelnames[i] = labelnames[i].decode("utf-8") 
        
    setname = 'train'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
    labelList = DATA[b'fine_labels']
    total_num = len(labelList)
    train_samples_per_cls = get_img_num_per_cls(no_of_classes, total_num, indices, 'exp', args.imb_factor)
    train_few_classes = torch.where(torch.tensor(train_samples_per_cls)<args.eta)[0]
    train_imgList, train_labelList = gen_imbalanced_data(train_samples_per_cls, imgList, labelList)
    datasets[setname] = CIFAR100LT(
        imageList=train_imgList, labelList=train_labelList, labelNames=labelnames,
        set_name=setname, isAugment=setname=='train')
    print('#examples in {}-set:'.format(setname), datasets[setname].current_set_len)

    setname = 'test'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
    labelList = DATA[b'fine_labels']
    image_count_per_class = Counter(labelList)
    if begin_class != -1:
        test_samples_per_cls = [image_count_per_class[idx] if idx in indices else 0 for idx in range(no_of_classes)]
        test_imgList, test_labelList = gen_imbalanced_data(test_samples_per_cls, imgList, labelList)
    else:
        test_imgList, test_labelList = imgList, labelList
    datasets[setname] = CIFAR100LT(
        imageList=test_imgList, labelList=test_labelList, labelNames=labelnames,
        set_name=setname, isAugment=setname=='train')
    print('#examples in {}-set:'.format(setname), datasets[setname].current_set_len)

    dataloaders = {set_name: DataLoader(datasets[set_name],
                                        batch_size=args.batch_size,
                                        shuffle=set_name=='train', 
                                        num_workers=os.cpu_count() // 4) # num_work can be set to batch_size
                   for set_name in ['train', 'test']}

    print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))
    return dataloaders, train_samples_per_cls, train_few_classes, train_labelList

def load_cifar10_batch(file):
    with open(file, 'rb') as obj:
        batch = pickle.load(obj, encoding='bytes')
    imgList = batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32))
    labelList = batch[b'labels']
    return imgList, labelList

def prepare_data_cifar10(args, no_of_classes, begin_class=-1, class_step=0):
    path_to_DB = './datasets'
    if not os.path.exists(path_to_DB): os.makedirs(path_to_DB)

    _ = torchvision.datasets.CIFAR10(root=path_to_DB, train=True, download=True)

    indices = range(no_of_classes)
    if begin_class != -1:
        if class_step == 0:
            indices = [begin_class]
        else:
            indices = range(begin_class, no_of_classes, class_step)

    path_to_DB = path.join(path_to_DB, 'cifar-10-batches-py')

    datasets = {}
    dataloaders = {}

    setname = 'batches.meta'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        labelnames = pickle.load(obj, encoding='bytes')
        labelnames = labelnames[b'label_names']
    for i in range(len(labelnames)):
        labelnames[i] = labelnames[i].decode("utf-8") 
        
    setname = 'train'
    train_imgList = []
    train_labelList = []
    for i in range(1, 6):  # 5 batches in CIFAR10
        file = os.path.join(path_to_DB, f'data_batch_{i}')
        imgList, labelList = load_cifar10_batch(file)
        train_imgList.append(imgList)
        train_labelList.append(labelList)
    imgList = np.concatenate(train_imgList)
    labelList = np.concatenate(train_labelList)
    
    total_num = len(labelList)
    train_samples_per_cls = get_img_num_per_cls(no_of_classes, total_num, indices, 'exp', args.imb_factor)
    train_few_classes = torch.where(torch.tensor(train_samples_per_cls)<args.eta)[0]
    train_imgList, train_labelList = gen_imbalanced_data(train_samples_per_cls, imgList, labelList)
    datasets[setname] = CIFAR10LT(
        imageList=train_imgList, labelList=train_labelList, labelNames=labelnames,
        set_name=setname, isAugment=setname=='train')
    print('#examples in {}-set:'.format(setname), datasets[setname].current_set_len)


    setname = 'test'
    test_file = os.path.join(path_to_DB, 'test_batch')
    imgList, labelList = load_cifar10_batch(test_file)
    image_count_per_class = Counter(labelList)
    if begin_class != -1:
        test_samples_per_cls = [image_count_per_class[idx] if idx in indices else 0 for idx in range(no_of_classes)]
        test_imgList, test_labelList = gen_imbalanced_data(test_samples_per_cls, imgList, labelList)
    else:
        test_imgList, test_labelList = imgList, labelList
    datasets[setname] = CIFAR10LT(
        imageList=test_imgList, labelList=test_labelList, labelNames=labelnames,
        set_name=setname, isAugment=setname=='train')
    print('#examples in {}-set:'.format(setname), datasets[setname].current_set_len)

    dataloaders = {set_name: DataLoader(datasets[set_name],
                                        batch_size=args.batch_size,
                                        shuffle=set_name=='train', 
                                        num_workers=os.cpu_count() // 4) # num_work can be set to batch_size
                   for set_name in ['train', 'test']}

    print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))
    return dataloaders, train_samples_per_cls, train_few_classes, train_labelList

if __name__ == '__main__':
    args = get_parser()
    main(args)
