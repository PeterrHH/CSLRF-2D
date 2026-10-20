import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import modules
import warnings
import torch.nn.functional as F
from modules.sync_batchnorm import convert_model
warnings.filterwarnings("ignore")
import utils
from modules.resnet import ResNet,BasicBlock
# class Get_Correlation(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         reduction_channel = channels//16
#         self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

#         self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
#         self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,1,1), groups=reduction_channel)
#         self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,2,2), dilation=(1,2,2), groups=reduction_channel)
#         self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,3,3), dilation=(1,3,3), groups=reduction_channel)
#         self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
#         self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
#         self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

#     def forward(self, x):

#         x2 = self.down_conv2(x)
#         affinities = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2))  # repeat the last frame
#         affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2))  # repeat the first frame 
#         features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2), F.sigmoid(affinities)-0.5 )* self.weights2[0] + \
#             torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2), F.sigmoid(affinities2)-0.5 ) * self.weights2[1] 

#         x = self.down_conv(x)
#         aggregated_x = self.spatial_aggregation1(x)*self.weights[0] + self.spatial_aggregation2(x)*self.weights[1] \
#                     + self.spatial_aggregation3(x)*self.weights[2]
#         aggregated_x = self.conv_back(aggregated_x)

#         return features * (F.sigmoid(aggregated_x)-0.5)
        
        
        
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         #print(f"INPLANES is {inplanes} and planes is {planes}")
#         self.conv1 = self.conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = self.conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#     def conv3x3(self,in_planes, out_planes, stride=1):
#     # 3x3x3 convolution with padding
#         return nn.Conv3d(
#             in_planes,
#             out_planes,
#             kernel_size=(1,3,3),
#             stride=(1,stride,stride),
#             padding=(0,1,1),
#             bias=False)


#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out
    
# class ResNet_Feat_Extract(nn.Module):

#     def __init__(self, block, layers, num_classes = 1000):
#         self.inplanes = 64
#         super(ResNet_Feat_Extract, self).__init__()

#         self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
#                                bias=False)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.corr1 = Get_Correlation(self.inplanes)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.corr2 = Get_Correlation(self.inplanes)
#         self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.corr3 = Get_Correlation(self.inplanes)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         # self.fc = Identity()

        
            
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

        
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv3d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=(1,stride,stride), bias=False),
#                 nn.BatchNorm3d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         N, C, T, H, W = x.size()
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         #print(f"x before layer 1 {x.shape}")
#         x = self.layer1(x)
#         #print(f"x before layer 2 {x.shape}")
#         x = self.layer2(x)
#         #print(f"x after layer 2 {x.shape}")
#         # corr_1 = 
#         # #print(f"corr 1 is {corr_1.shape}")
#         x = x + self.corr1(x) * self.alpha[0]
#         x = self.layer3(x)
#         x = x + self.corr2(x) * self.alpha[1]
#         x = self.layer4(x)
#         x = x + self.corr3(x) * self.alpha[2]
#         x = x.transpose(1,2).contiguous() # (batch_size,time, 256,7,7)
#         bc,time,channel,h,w = x.shape
#         x = x.view(bc*time,channel,h,w)
#         x = self.avgpool(x)
#         x = x.view(bc,time,channel,1,1) # batch_size, time, emebdding)
#         x = x.squeeze()
#         if x.dim() == 2:
#             x = x.unsqueeze(0)
#         # print(f"X shape after squeezing {x.shape}")
#         # x = x.view(-1,channel)
#         # print(f"X shape after view {x.shape}")
#         # x = self.fc(x)
#         # print(f"X shape after fc {x.shape}")
#         # x = x.view(bc, time, channel)
#         return x
        
# embedding size is 512
class CPC(nn.Module):
    def __init__(self, n_pred_step,embedding_dim, context_dim , hidden_dim,num_layers, device, num_classes = 1296):
        super(CPC, self).__init__()
        self.n_pred_step = n_pred_step
        self.embedding_dim = embedding_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
 
        '''
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        self.linear_vid = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.video_predictors = nn.ModuleList([
            nn.Linear(self.context_dim, self.embedding_dim) for _ in range(self.n_pred_step)
        ])
        self.ar = nn.LSTM(self.embedding_dim, self.context_dim, self.num_layers, batch_first = True)# LSTM OUTPUT
        self.initial_extractor = self.load_initial_extractor()
        #self.res_lin = nn.Linear(3*224*224,self.embedding_dim)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        '''
        self.initial_extractor = self.model_to_device(self.load_initial_extractor())
        self.softmax = self.device.model_to_device(nn.Softmax())
        self.lsoftmax = self.device.model_to_device(nn.LogSoftmax())
        # self.cnn_encoder = 
        self.linear_vid = self.model_to_device(nn.Linear(self.embedding_dim,self.embedding_dim))
        self.video_predictors = nn.ModuleList([
            self.model_to_device(nn.Linear(self.context_dim, self.embedding_dim)) for _ in range(self.n_pred_step)
        ])
        self.ar = self.model_to_device(nn.LSTM(self.embedding_dim, self.context_dim, self.num_layers, batch_first = True)) # LSTM OUTPUT
        self.res_lin = self.model_to_device(nn.Linear(3*224*224,self.embedding_dim))

        #self.loss = loss
    def load_initial_extractor(self):
        model = ResNet_Feat_Extract(BasicBlock, [2, 2, 2, 2])
        checkpoint = model_zoo.load_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        layer_name = list(checkpoint.keys())
        for ln in layer_name :
            if 'conv' in ln or 'downsample.0.weight' in ln:
                checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
        model.load_state_dict(checkpoint, strict=False)
        # model.eval()
        return model
    
    def model_to_device(self,model):
        #print(f"MMODEL IS TO {self.device.output_device} GPU LIst {self.device.gpu_list}")
        #model = model.cuda(self.device.output_device)
        model = model.to(self.device.output_device)
        
        if len(self.device.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.device.gpu_list)
        return model
    
    def forward(self,video):
        '''
        predict_video_encode: video embedding needed to be predicted (pred_size, batch_size, embedding_size)
        video_forward: all frame embedding up to t. (batch_size, t, embedding_size)
        video_context is the c_t (batch_size, context dim)
        '''
        #print(f"IN CPC NOW with input shape {video.shape}")
        #    input x size (batch_size, time, 3, img_size, img_size)

        video = video.permute(0,2,1,3,4) # permute to (batch, 3, time, img_size, img_size)
        
        batch_size, _, time_length,_ , _= video.shape
        t_sample = torch.randint(self.n_pred_step+1, time_length - self.n_pred_step, (1,)).long()

        #video_encodings = self.initial_extractor(video).to(self.device) # (batch_size, time, emb_size = 512)
        #video_encodings = self.linear_vid(video_encodings).to(self.device)

        video_encodings = self.initial_extractor(video)# (batch_size, time, emb_size = 512)
        #video_encodings = self.linear_vid(video_encodings)

        '''
        video = video.reshape(batch_size * time_length, -1)
        video_encodings = self.res_lin(video)
        video_encodings = video_encodings.reshape(batch_size, time_length,self.embedding_dim)
        '''
        
        pred_video_encode = torch.empty((self.n_pred_step, batch_size, self.embedding_dim)) # (num_pred_step, batch_size, embedding)
        for i in range(1, self.n_pred_step + 1):
            pred_video_encode[i - 1] = video_encodings[:, t_sample + i, :].reshape(batch_size,
                                                                    self.embedding_dim)  # zat pred[i-1]= shape (batch_size, embedding)
        #print(f"predict video encode {pred_video_encode.shape}")
        video_forward = video_encodings[:, :t_sample+1, :]  # e.g. size batch_size, time from 0 to t, 512
        #print(f"t sample {t_sample} and shape is {video_forward.shape}")
        # Autoregressive network
        #video_forward = video_forward.to(self.device)
        #video_hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
         #               torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
       # print(f"forward shape is {video_forward.shape} hidden shape is {video_hidden[0].shape}")
    
        # video_hidden = self.device.data_to_device((torch.zeros(self.num_layers, batch_size, self.context_dim),
        #                 torch.zeros(self.num_layers, batch_size, self.context_dim)))
       #video_hidden = (torch.zeros(self.num_layers, batch_size, self.context_dim),
         #       torch.zeros(self.num_layers, batch_size, self.context_dim))
           
        video_context,_ = self.ar(video_forward)
        video_context_full,_ = self.ar(video_encodings)
        video_context = video_context[:, t_sample, :].reshape(batch_size, self.context_dim)  
        
        # video_pred = torch.empty((self.n_pred_step, batch_size, self.embedding_dim)).to(self.device) # e.g. size 12*8*512
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
            #correct = torch.sum(torch.eq(total_max,torch.arange(0, batch_size)))
            correct = torch.sum(torch.eq(total_max,self.device.data_to_device(torch.arange(0, batch_size))))
            
#             print(f"-------------------- i = {i} -------------------- ")
#             print(f"encode shape {pred_video_encode[i].shape} and prediction is {video_pred[i].shape}")
#             print(total)
#             print("Total\n")
#             print(self.softmax(total))
#             print("argmax\n")
#             print(torch.argmax(self.softmax(total), dim=0))
#             print("arrange\n")
#             print(torch.arange(0,batch_size))
            
#             print(f"correct = {correct}")
#             print(f"-------------------- i = {i} -------------------- ")
            
            w = 1.0
            nce_loss += w * torch.sum(torch.diag(self.lsoftmax(total)))
        nce_loss /= -1. * batch_size * self.n_pred_step
        acc = 1. * correct / batch_size

        #print(f"LOSS IS {nce_loss}")
        #print(f"ACCURACY IS {acc} with correct is {correct}")
        return nce_loss,acc, video_encodings, video_context_full
            
    def inference(self,video):
        
        video = video.permute(0,2,1,3,4) # permute to (batch, 3, time, img_size, img_size)
    
        batch_size, _, time_length,_ , _ = video.shape
        #t_sample = torch.randint(self.n_pred_step+1, time_length - self.n_pred_step, (1,)).long()

        # video_encodings = self.initial_extractor(video).to(self.device) # (batch_size, time, emb_size = 512)
        # video_encodings = self.linear_vid(video_encodings).to(self.device)
        
        video_encodings = self.initial_extractor(video)# (batch_size, time, emb_size = 512)
        video_encodings = self.linear_vid(video_encodings)
       

        #video = video.reshape(batch_size * time_length, -1)
        #video_encodings = self.res_lin(video)
        #video_encodings = video_encodings.reshape(batch_size, time_length,self.embedding_dim)
       #
        video_context,_ = self.ar(video_encodings)
        
        # video_context = video_context[:, t_sample, :].reshape(batch_size, self.context_dim) 
        #print(f"encoding shape {video_encodings.shape} context shape {video_context.shape}")
        return video_encodings.permute(0, 2, 1), video_context.permute(0, 2, 1)
    
    
class ResNet_LSTM(nn.Module):
    
    def __init__(self, n_pred_step,embedding_dim, context_dim , hidden_dim,num_layers, device):
        super(ResNet_LSTM, self).__init__()
        self.n_pred_step = n_pred_step
        self.embedding_dim = embedding_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers


        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        self.linear_vid = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.video_predictors = nn.ModuleList([
            nn.Linear(self.context_dim, self.embedding_dim) for _ in range(self.n_pred_step)
        ])
        self.linear_vid = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.ar = nn.LSTM(self.embedding_dim, self.context_dim, self.num_layers, batch_first = True)# LSTM 
        self.initial_extractor = self.load_initial_extractor()
            
    def load_initial_extractor(self):
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        checkpoint = model_zoo.load_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        layer_name = list(checkpoint.keys())
        for ln in layer_name :
            if 'conv' in ln or 'downsample.0.weight' in ln:
                checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
        model.load_state_dict(checkpoint, strict=False)
        # model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def forward(self,video):        
        batch, temp, channel, height, width = video.shape
        video = video.permute(0,2,1,3,4)

        #print(f"Before resnet shape {video.shape}")
        
        video_encodings = self.initial_extractor(video)# (batch_size, time, emb_size = 512)
        video_encodings = video_encodings.view(batch,temp, 512)
        #print(f"after resnet shape {video_encodings.shape}")
        video_encodings = self.linear_vid(video_encodings)
        #print(f"after linear shape {video_encodings.shape}")
        context,_ = self.ar(video_encodings)
        #print(f"out shape {out.shape}")
        
        return video_encodings, context
        
    def calc_loss(self,video_encodings):
        batch_size, time_length,_ = video_encodings.shape
        t_sample = torch.randint(self.n_pred_step+1, time_length - self.n_pred_step, (1,)).long()
        pred_video_encode = torch.empty((self.n_pred_step, batch_size, self.embedding_dim)) # (num_pred_step, batch_size, embedding)
        for i in range(1, self.n_pred_step + 1):
            pred_video_encode[i - 1] = video_encodings[:, t_sample + i, :].reshape(batch_size,
                                                                    self.embedding_dim)  # zat pred[i-1]= shape 
        video_forward = video_encodings[:, :t_sample+1, :]  # e.g. size batch_size, time from 0 to t, 512

        video_context,_ = self.ar(video_forward)
        video_context = video_context[:, t_sample, :].reshape(batch_size, self.context_dim)  

        # video_pred = torch.empty((self.n_pred_step, batch_size, self.embedding_dim)).to(self.device) # e.g. size 12*8*512
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
            # correct = torch.sum(torch.eq(total_max,self.device.data_to_device(torch.arange(0, batch_size))))

            w = 1.0
            nce_loss += w * torch.sum(torch.diag(self.lsoftmax(total)))
        nce_loss /= -1. * batch_size * self.n_pred_step
        acc = 1. * correct / batch_size

        #print(f"LOSS IS {nce_loss}")
        #print(f"ACCURACY IS {acc} with correct is {correct}")
        return nce_loss,acc

if __name__ == "__main__":
    
    video = torch.randn(8,80,3,224,224)
    res = ResNet_LSTM(2,512,512,512,3,None)
    out = res.forward(video)
    print(out.shape)
    loss,acc = res.calc_loss(out)
    print(f"loss {loss} acc {acc}")