import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.metrics 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms
from utils.eval_funcs import print_accuracy


def train_model(dataloaders, model, lossFunc, 
                optimizer, scheduler,
                num_epochs=50, model_name= 'CE', work_dir='./work_dir', device='cuda', print_each = 1, optim_mode='sgd', tail_classes=None, 
                train_labelList = None, reweight=1, imb_factor=50, dataset='cifar100'):

    since = time.time()
    best_perClassAcc = 0.0
    
    phases = ['train', 'test']
    
    for epoch in range(num_epochs):  
        if epoch%print_each==0:
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            predList = np.array([])
            grndList = np.array([])
            
            if phase == 'train':
                print('train')
                scheduler[0].step()                
                if len(scheduler) == 3:
                    scheduler[1].step()
                    scheduler[2].step()
                model.train()
            else:
                if epoch % 10 != 9:
                    continue
                print('test')
                model.eval()  # Set model to training mode  
              
            running_loss = 0.0
            running_acc = 0.0
            
            # Iterate over data.
            iterCount, sampleCount = 0, 0
            for sample in dataloaders[phase]:                
                images, targets = sample
                images = images.to(device)
                targets = targets.type(torch.long).view(-1).to(device)

                with torch.set_grad_enabled(phase=='train'):
                    logits = model(images)
                    loss = lossFunc(logits, targets)
                    softmaxScores = logits.softmax(dim=1)

                    preds = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)                  
                    accRate = (targets.type(torch.float).squeeze() - preds.squeeze().type(torch.float))
                    accRate = (accRate==0).type(torch.float).mean()
                    
                    predList = np.concatenate((predList, preds.cpu().numpy()))
                    grndList = np.concatenate((grndList, targets.cpu().numpy()))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = forbackward(model, lossFunc, optimizer, optim_mode, tail_classes, images, targets, loss)
                    else:
                        loss = loss.mean()
                        
                # statistics  
                iterCount += 1
                sampleCount += targets.size(0)
                running_acc += accRate*targets.size(0) 
                running_loss += loss.item() * targets.size(0) 
                
                print2screen_avgLoss = running_loss / sampleCount
                print2screen_avgAccRate = running_acc / sampleCount
                
            epoch_error = print2screen_avgLoss      
            
            confMat = sklearn.metrics.confusion_matrix(grndList, predList)                
            # normalize the confusion matrix
            a = confMat.sum(axis=1).reshape((-1,1))
            confMat = confMat / a
            curPerClassAcc = 0
            for i in range(confMat.shape[0]):
                curPerClassAcc += confMat[i,i]
            curPerClassAcc /= confMat.shape[0]
            if epoch%print_each==0:
                print('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}'.format(
                    epoch_error, print2screen_avgAccRate, curPerClassAcc))

            # if (epoch+1) % 2 == 0 and (epoch + 1) != num_epochs and phase == 'test':
            if (epoch+1) % 2 == 0 and (epoch + 1) != num_epochs:
                print_accuracy(model, dataloaders, train_labelList, epoch=epoch+1, dataset=dataset, device=device, save_dir=work_dir)
                # save model of current epoch
                path_to_save_param = os.path.join(work_dir, model_name+'_best_'+str(imb_factor)+
                                                  '_epoch'+str(epoch+1)+'.pth')
                torch.save(model.state_dict(), path_to_save_param)
                print('Current model params have been successfully saved to '+path_to_save_param)
            if (phase=='val' or phase=='test') and curPerClassAcc>best_perClassAcc: 
                best_perClassAcc = curPerClassAcc

                path_to_save_param = os.path.join(work_dir, model_name+'_best_'+str(imb_factor)+'.pth')
                torch.save(model.state_dict(), path_to_save_param)

            # Update rho if ssesam
            if optim_mode == 'ssesam' and phase == 'train':
                optimizer.update_rho()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    

def forbackward(model, lossFunc, optimizer, optim_mode, tail_classes, images, targets, loss):
    if optim_mode == 'sgd':
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
    elif optim_mode == 'sam':
        loss = loss.mean()
        loss.backward()
        optimizer.first_step()
        
        logits = model(images)
        loss = lossFunc(logits, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.second_step()
        
    elif optim_mode == 'imbsam':
        tail_mask = torch.where((targets[:, None] == tail_classes[None, :].to(targets.device)).sum(1) == 1, True, False)
        head_count = targets.size(0) - tail_mask.sum().item()
        tail_count = tail_mask.sum().item()
                            
        # head_loss = loss[~tail_mask].sum() / targets.size(0) 
        head_loss = loss[~tail_mask].sum() / head_count
        head_loss.backward(retain_graph=True)
        optimizer.first_step()
                            
        tail_loss = loss[tail_mask].sum() / tail_count
        tail_loss.backward()
        optimizer.second_step()
                            
        logits = model(images)
        tail_loss = lossFunc(logits[tail_mask], targets[tail_mask]).sum() / tail_count
        tail_loss.backward()
        optimizer.third_step()

        # loss = head_loss + tail_loss
        loss = (head_loss * head_count + tail_loss * tail_count) / len(targets)
        
    elif optim_mode == 'ssesam':
        tail_mask = torch.where((targets[:, None] == tail_classes[None, :].to(targets.device)).sum(1) == 1, True, False)
        head_count = targets.size(0) - tail_mask.sum().item()
        tail_count = tail_mask.sum().item()
        class_sizes = [head_count, tail_count]
                            
        head_loss = loss[~tail_mask].sum() / targets.size(0)
        head_loss.backward(retain_graph=True)
        optimizer.compute_and_add_epsilon(n_i=0)

        logits = model(images)
        head_loss = lossFunc(logits[~tail_mask], targets[~tail_mask]).sum() / targets.size(0)
        head_loss.backward(retain_graph=True)
        optimizer.compute_grad_sum_and_restore_p()
                            
        logits = model(images)
        tail_loss = lossFunc(logits[tail_mask], targets[tail_mask]).sum() / targets.size(0)
        tail_loss.backward(retain_graph=True)
        optimizer.compute_and_add_epsilon(n_i=1)
                            
        logits = model(images)
        tail_loss = lossFunc(logits[tail_mask], targets[tail_mask]).sum() / targets.size(0)
        tail_loss.backward(retain_graph=True)
        optimizer.compute_grad_sum_and_restore_p()

        optimizer.update()

        loss = head_loss + tail_loss
 
    return loss
