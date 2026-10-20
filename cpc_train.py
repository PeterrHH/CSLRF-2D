import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
import gc
'''

Input a CPC model, a optimizer, device, which epoch it is. 
'''
def cpc_pretrain(loader, model,optimizer, device, epoch_idx):
    #print(f"In the pretraining function with device {device.output_device}")
    loss_value = []
    accuracy_value = []
    #print(f"Start Pretraining with device {device}")
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0]).cuda()
        # vid_lgt = device.data_to_device(data[1])
        optimizer.zero_grad()

        with autocast():
            framewise, context= model(vid)
            loss, acc = model.calc_loss(framewise)
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
        accuracy_value.append(acc.item())
        
        # print(
        #     '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f} acc: {:.6}'
        #         .format(epoch_idx, batch_idx, len(loader), str(loss.item()), str(acc.item()))
       # print(f"Epoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. Loss: {loss.item()}, acc: {acc.item()}")
        # gc.collect()
        torch.cuda.empty_cache()
        del acc
        del loss
    # optimizer.scheduler.step()
    # print('\tMean training loss: {:.10f}. Mean accuracy : {:.6f}'.format(np.mean(loss_value),np.mean(accuracy)))
    print(f"Mean training loss: {np.mean(loss_value)} Mean accuracy {np.mean(accuracy_value)}. len is {len(loss_value)}")
    return np.mean(loss_value), np.mean(accuracy_value)

def train_with_cpc(loader, model, optimizer, device, epoch_idx, recoder):
    
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    #print("gets here")
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            loss = model.criterion_calculation(ret_dict, label, label_lgt)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            #print(data[-1])
            print(str(data[1])+'  frames')
            print(str(data[3])+'  glosses')
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del ret_dict
        del loss
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return np.mean(loss_value)


