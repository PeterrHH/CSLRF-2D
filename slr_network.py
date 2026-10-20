import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv, EncoderLayer
import modules.resnet as resnet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, n_pred_step, pixel_pred_step, embedding_dim, context_dim, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.n_pred_step = n_pred_step
        self.time_pred_step = n_pred_step
        self.pixel_pred_step = pixel_pred_step
        self.loss_weights = loss_weights
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        self.linear_vid = nn.Linear(512,512)
        self.ar = nn.LSTM(512, 512, 3, batch_first = True)# LSTM 
        #self.conv2d = None
        
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True) # LSTM/GRU
        
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        self.video_predictors = nn.ModuleList([
            nn.Linear(self.context_dim, self.embedding_dim) for _ in range(self.n_pred_step)
        ])
        
        self.att = EncoderLayer(d_model = 1024, nhead = 1, dim_feedforward = 1024, dropout = 0.1)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        
        #self.register_backward_hook(self.backward_hook)
        
#         self.video_predictors = nn.ModuleList([
#             nn.Linear(self.context_dim, self.embedding_dim) for _ in range(self.n_pred_step)
#         ])
        
        self.cpc = None
        self.res_lstm_use = None
        self.res_lstm = None

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0
            
    def set_cpc(self, model, emb_use):
        
        self.cpc = model
        self.emb_use = emb_use
        
    def set_res_lstm(self,model):
        self.res_lstm = model
        self.res_lstm_use = True

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x
    
    def calc_CPC_loss(self, video_encodings,device):
        #print(f"Input video_encodings shape at CPC loss {video_encodings.shape}")
        '''
        batch_size, _, time_length = video_encodings.shape
        video_encodings = video_encodings.permute(0,2,1)
        t_sample = torch.randint(self.n_pred_step+1, time_length - self.n_pred_step, (1,)).long()
        pred_video_encode = torch.empty((self.n_pred_step, batch_size, self.embedding_dim))
        #print(f"pred_video_encoding {pred_video_encode.shape}")
        for i in range(1, self.n_pred_step + 1):
            pred_video_encode[i - 1] = video_encodings[:, t_sample + i, :].reshape(batch_size,
                                                                    self.embedding_dim)  # zat pred[i-1]= shape 
        video_forward = video_encodings[:, :t_sample+1, :]
        video_context,_ = self.ar(video_forward)
        video_context = video_context[:, t_sample, :].reshape(batch_size, self.context_dim)
        
        video_pred = torch.empty((self.n_pred_step, batch_size, self.embedding_dim))

        for i in range(0, self.n_pred_step):
            video_linear = self.video_predictors[i]
            video_pred[i] = video_linear(video_context)  # Wk*c_t e.g. size 8*512
        
        # Compute Loss between Video context prediction and the actual feature extracted from future frame
        nce_loss = 0
        for i in range(0,self.n_pred_step):
            

            total = torch.mm(pred_video_encode[i], torch.transpose(video_pred[i], 0, 1))# W*c_t

            #total_max = torch.argmax(self.softmax(total),dim = 0).to(self.device)
            #correct = torch.sum(torch.eq(total_max,torch.arange(0, batch_size).to(self.device)))
            total_max = torch.argmax(self.softmax(total),dim = 0)
            correct = torch.sum(torch.eq(total_max,torch.arange(0, batch_size)))
            w = 1.0
            nce_loss += w * torch.sum(torch.diag(self.lsoftmax(total)))
        nce_loss /= -1. * batch_size * self.n_pred_step
        acc = 1. * correct / batch_size
        '''
       
        batch_size, time_length, emb, img_size, img_size = video_encodings.shape
        video = video_encodings.reshape(batch_size, time_length, emb, img_size*img_size)
        video = video.permute(0,1,3,2)
        video = video.view(-1, 49,512)

        video = self.linear_vid(video) # Linear output of the 
       
        video = video.view(batch_size, time_length, 49, 512)
        t_sample = torch.randint(self.time_pred_step+1, time_length - self.time_pred_step, (1,)).long()
        pixel_sample = torch.randint(self.pixel_pred_step+1,49-self.pixel_pred_step,(1,)).long()

        pred_video_encode = torch.empty((self.time_pred_step, self.pixel_pred_step, batch_size, self.embedding_dim))

        for t in range(1, self.time_pred_step + 1):
            for p in range(1, self.pixel_pred_step+1):
                pred_video_encode[t - 1][p-1] = video[:, t_sample + t, pixel_sample + p,:].reshape(batch_size,
                                                                    self.embedding_dim)  # zat pred[i-1]= shape 
    
        video_forward = video[:, :t_sample+1, :pixel_sample+1, :]

        batch_size,forward_time, pixel_num, emb = video_forward.shape
        video_forward = video_forward[:,t_sample,:pixel_sample,:]# pick the t_sample as the time (batch_size,1, 0-num_pixel, embedding)
        video_forward = torch.squeeze(video_forward,dim = 1)# (batch_size, 0-pixel, embedding)
        
        video_context,_ = self.ar(video_forward) # Output (batch_size, time, embedding) for pixel at p_pixel

        video_context = torch.squeeze(video_context[:, -1, :]) # Get the context to (batch_size,embedding) this is the context for t_sample and for pixel at pixel_sample
        
        video_pred = torch.empty((self.time_pred_step, self.pixel_pred_step,batch_size, self.embedding_dim))

#         # reshape_context = video.view(-1,512*49)
#         for t in range(0, self.time_pred_step):
#             for p in range(0, self.pixel_pred_step): # Do we share the same linear for the saem time step?
#                 video_linear = self.video_predictors[t]
#                 video_pred[t][p] = video_linear(video_context)  # Wk*c_t e.g. size 8*512
               
     
        # calculate nce LOSS
        nce_loss = 0
        for t in range(0, self.time_pred_step):
            for p in range(0, self.pixel_pred_step):
            
                total = torch.mm(pred_video_encode[t][p], torch.transpose(video_pred[t][p], 0, 1))
                total_max = torch.argmax(self.softmax(total),dim = 0)
                correct = torch.sum(torch.eq(total_max,torch.arange(0, batch_size)))
                w = 1.0
                nce_loss += w * torch.sum(torch.diag(self.lsoftmax(total)))
        nce_loss /= -1. * batch_size * self.time_pred_step * self.pixel_pred_step
        acc = 1. * correct / batch_size
        #print(f"LOSS IS {nce_loss}")
        #print(f"ACCURACY IS {acc} with correct is {correct}")

        return nce_loss,acc
    
    def forward(self, x, len_x = None, label=None, label_lgt=None):
        cpc_loss = None
        # self.conv2d.eval()
        if self.cpc is None and self.res_lstm_use is None:
            if len(x.shape) == 5:
                # videos
                #print("Going into the rightplace")
                batch, temp, channel, height, width = x.shape
                #inputs = x.reshape(batch * temp, channel, height, width)
                #framewise = self.masked_bn(inputs, len_x)
                #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
                pool =  self.conv2d(x.permute(0,2,1,3,4))
              
                framewise = pool
                #framewise = framewise.view(batch, temp, -1).permute(0,2,1) # btc -> bct
                #print(f"SELF CONV2DING")
                #print(f"framewise here below lin and lstm is {framewise_out.shape}")
            else:
                # frame-wise features
                framewise_out = x
        elif self.cpc is not None:
            embedding ,context = self.cpc(x) # Freeze
            #print(f"AFTER CPC embedding shape {embedding.shape} and context shape {context.shape}")
            if self.emb_use == "embedding":
                framewise = embedding
            elif self.emb_use == "context":
                framewise = context
            else:
                raise ValueError("Emebdding definition after CPC is incorrect")
            framewise = framewise.permute(0,2,1)
        #print(f"Before conv1d, frame wise should shape {framewise.shape} and lenx is {len_x}")
        else :
            framewise_out = self.res_lstm(x)
            framewise_out = framewise_out.permute(0,2,1)
        '''
        1D case
        #print(f"framewise out shape {framewise_out.shape}")
        # #print(f"framewise device {framewise_out.get_device()}")
        # framewise = framewise.permute(0,2,1)
        # #print(f"Frame after ResNet {framewise.shape}")
        # framewise = self.linear_vid(framewise)
        # #print(f"Frame after Lin{framewise.shape}")
        # framewise,_ = self.ar(framewise)
        
        # framewise = framewise.permute(0,2,1)
        
        '''
        
    
        #2D Case
        #print(f"out of the gate shape {framewise.shape}")

        bs,t,emb,feat_size,feat_size = framewise.shape
        framewise = framewise.reshape(bs,t,feat_size*feat_size,emb)
        framewise = framewise.view(bs*t,feat_size*feat_size,emb)
        framewise,_ = self.ar(framewise)
        framewise = framewise.reshape(bs*t,emb,feat_size,feat_size)
        framewise = self.conv2d.avgpool(framewise)

        
        if bs == 1:
            framewise = framewise.reshape(bs,t,emb,1,1).squeeze().unsqueeze(dim = 0).permute(0,2,1)
        else:
            framewise = framewise.reshape(bs,t,emb,1,1).squeeze().permute(0,2,1)
        #print(framewise.shape)
        
        
        #print(f"framewise shape is {framewise.shape}")
        # frame size expect([bs, emb, time])
        #print(f"framewise shape {framewise.shape}")
        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        #print(f"After Conv 1d")
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
       
        # Add self attention (time,batch,embedding) 

        x.permute(0,2,1)
        x = self.att(x)
  

        tm_outputs = self.temporal_model(x, lgt) # pass into LSTM Decoder
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": pool,
            # "visual_features": x,
            "feat_len": lgt.cuda(),
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }
        

    def criterion_calculation(self, ret_dict, label, label_lgt,device, compute_cpc):
        #print(f"Calculating LOSS conv logits {ret_dict['sequence_logits'].shapes}")
        
        loss = 0
        cpc_loss = 0
        if compute_cpc is True:
            # cpc_loss,_ = self.calc_CPC_loss(ret_dict["framewise_features"],device)
            cpc_loss, _ = self.calc_CPC_loss(ret_dict["framewise_features"],device)

        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        if compute_cpc:
            loss += 3*cpc_loss
        print(f"LOSS IS {loss}")
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8) # Change this as well
        return self.loss

    

