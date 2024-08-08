import os
import json
import random
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from mh_models import PATvidx4
from mh_utils import MultiSeqSet
from utils import AverageMeter, L1Loss, cal_psnr, save_ckpt, weights_init_xavier
from tensorboardX import SummaryWriter

from tqdm import tqdm


def trainer(cfg):
    
    outputs_dir = os.path.join('log', cfg.outputs_dir)
    print('[*] Saving outputs to {}'.format(outputs_dir))
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        
    logs_dir = os.path.join('ckpt', cfg.logs_dir)
    print('[*] Saving tensorboard logs to {}'.format(logs_dir))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    writer = SummaryWriter(logs_dir)
    
    net = PATvidx4().to(cfg.device)
    net = nn.DataParallel(net)
    if cfg.initial_weight is None:
        net.apply(weights_init_xavier)
    else:
        print("Loading weights from {:s}".format(cfg.initial_weight))
        net.load_state_dict(torch.load(cfg.initial_weight)['state_dict'])
    cudnn.benchmark = True

    if cfg.loss_type.lower()=='mse' or cfg.loss_type.lower()=='l2':
        criterion = nn.MSELoss().to(cfg.device)
    elif cfg.loss_type.lower()=='l1':
        criterion = nn.L1Loss().to(cfg.device)
    else:
        raise RuntimeError("Unsupported loss: {}".format(cfg.loss_type))
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    
    train_set = MultiSeqSet(cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_set = MultiSeqSet(cfg.validset_dir)
    valid_loader = DataLoader(dataset=valid_set, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=False, drop_last=False)
    print("Training sets from: {}".format(cfg.trainset_dir))
    print("Validation sets from: {}".format(cfg.validset_dir))

    best_psnr = 0.0
    for idx_epoch in range(cfg.n_epochs):
        train(idx_epoch, net, train_loader, criterion, optimizer, cfg.device, writer)
        val_psnr = valid(idx_epoch, net, valid_loader, cfg.device, writer)
        scheduler.step()

        if val_psnr >= best_psnr:
            print("[*]")
            best_psnr = val_psnr
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'psnr': val_psnr,
            }, save_path = outputs_dir, filename='best.pth.tar')
            
def train(epoch, model, train_loader, criterion, optimizer, device, writer):
    model.train()
    loss_epoch = AverageMeter()
    
    with tqdm(total=len(train_loader)) as t:
        t.set_description(f'epoch {epoch+1}')
        for idx_iter, (in_tuple, gt) in enumerate(train_loader):
            # go through network
            in_tuple = [x.to(device) for x in in_tuple]
            gt = gt.to(device)
            lr_ins, hr_in, poss = in_tuple[:3], in_tuple[3], in_tuple[4:]
            pred, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = model(lr_ins, hr_in, 1, poss)
            
            # loss
            loss = criterion(gt, pred)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.update(loss.item())
            t.set_postfix(loss='{:.6f}'.format(loss_epoch.avg))
            t.update(1)
    
    writer.add_scalar('Stats/training_loss', loss_epoch.avg, epoch+1)

def valid(epoch, model, valid_loader, device, writer):
    model.eval()
    psnr_epoch = AverageMeter()
    
    with tqdm(total=len(valid_loader)) as t:
        t.set_description('validate')
        for idx_iter, (in_tuple, gt) in enumerate(valid_loader):
            # go through network
            in_tuple = [x.to(device) for x in in_tuple]
            gt = gt.to(device)
            lr_ins, hr_in, poss = in_tuple[:3], in_tuple[3], in_tuple[4:]
            with torch.no_grad():
                pred, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
                (V_left_to_right, V_right_to_left) = model(lr_ins, hr_in, 1, poss)

            psnr_epoch.update(cal_psnr(gt.data.cpu(), pred.data.cpu()))

            t.set_postfix(psnr='{:.2f}'.format(psnr_epoch.avg))
            t.update(1)
            
    writer.add_scalar('Stats/valid_psnr', psnr_epoch.avg, epoch+1)

    return psnr_epoch.avg

def main(cfg):
    trainer(cfg)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=40, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='/groups/djbrady/minghao/DAVIS_PAT_trainval/train/')
    parser.add_argument('--validset_dir', type=str, default='/groups/djbrady/minghao/DAVIS_PAT_trainval/valid/')
    parser.add_argument('--outputs_dir', type=str, default='vid')
    parser.add_argument('--logs_dir', type=str, default='vid')
    parser.add_argument('--initial_weight', type=str, default=None)
    parser.add_argument('--loss_type', type=str, default='mse')    
    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

