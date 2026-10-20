'''
Input shape is (bc, time, emb = 512, 7,7)
reshape it to 1D at 49
Linear layer to the whole thing
get a time example
run RNN on the time example
'''
import torch
import torch.nn as nn

class twoD_CPC(nn.Module):
    def __init__(self, time_pred_step, pixel_pred_step):
        super().__init__()
        self.time_pred_step = time_pred_step
        self.pixel_pred_step = pixel_pred_step
        self.embedding_dim = 512
        self.context_dim = 512
        self.lin = nn.Linear(512,512)
        self.ar = nn.LSTM(512, 512, 3, batch_first = True)# LSTM
        self.video_predictors = nn.ModuleList([
            nn.Linear(self.context_dim, self.embedding_dim) for _ in range(self.time_pred_step)
        ])
                
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        pass
    
    def forward(self, video):  # Input should already passed through ResNet
        batch_size, time_length, emb, img_size, img_size = video.shape
        video = video.reshape(batch_size, time_length, emb, img_size*img_size)
        print(video.shape)
        video = video.permute(0,1,3,2)
        video = video.view(-1, 49,512)
        print(f"Before lin {video.shape}")
        video = self.lin(video) # Linear output of the 
        print(f"Out is {video.shape}")
        video = video.view(batch_size, time_length, 49, 512)
        print(f"Shape is {video.shape}")
        t_sample = torch.randint(self.time_pred_step+1, time_length - self.time_pred_step, (1,)).long()
        pixel_sample = torch.randint(self.pixel_pred_step+1,49-self.pixel_pred_step,(1,)).long()
        print(f"time sample {t_sample} pixel sample {pixel_sample}")
        pred_video_encode = torch.empty((self.time_pred_step, self.pixel_pred_step, batch_size, self.embedding_dim))

        for t in range(1, self.time_pred_step + 1):
            for p in range(1, self.pixel_pred_step+1):
                pred_video_encode[t - 1][p-1] = video[:, t_sample + t, pixel_sample + p,:].reshape(batch_size,
                                                                    self.embedding_dim)  # zat pred[i-1]= shape 
        print(f"Pred VIdeo Encode overall shape {pred_video_encode.shape}")
    
        video_forward = video[:, :t_sample+1, :pixel_sample+1, :]
        print(f"Video forward shape {video_forward.shape}")
        batch_size,forward_time, pixel_num, emb = video_forward.shape
        video_forward = video_forward[:,:t_sample,pixel_sample,:]# pick the t_sample as the time (batch_size,1, 0-num_pixel, embedding)
        video_forward = torch.squeeze(video_forward,dim = 2)# (batch_size, 0-time, embedding)
        print(f"After rehsaping {video_forward.shape}") 

        
        video_context,_ = self.ar(video_forward) # Output (batch_size, time, embedding) for pixel at p_pixel
        print(f"Video context {video_context.shape}")
        video_context = torch.squeeze(video_context[:, -1, :]) # Get the context to (batch_size,embedding) this is the context for t_sample and for pixel at pixel_sample
        print(f"Video context after reshaping {video_context.shape}") # 
        
        video_pred = torch.empty((self.time_pred_step, self.pixel_pred_step,batch_size, self.embedding_dim))

        # reshape_context = video.view(-1,512*49)
        for t in range(0, self.time_pred_step):
            for p in range(0, self.pixel_pred_step): # Do we share the same linear for the saem time step?
                video_linear = self.video_predictors[t]
                video_pred[t][p] = video_linear(video_context)  # Wk*c_t e.g. size 8*512
               
        print(f"Video pred shape {video_pred.shape}")       
        '''
        Video pred now have shape (time_pred_step,pixel_pred_step,batch_size, embedding)
        '''
        
        
        # calculate nce LOSS
        nce_loss = 0
        for t in range(0, self.time_pred_step):
            for p in range(0, self.pixel_pred_step):
            
                total = torch.mm(pred_video_encode[t][p], torch.transpose(video_pred[t][p], 0, 1))
                print(total.shape)
                total_max = torch.argmax(self.softmax(total),dim = 0)
                correct = torch.sum(torch.eq(total_max,torch.arange(0, batch_size)))
                w = 1.0
                nce_loss += w * torch.sum(torch.diag(self.lsoftmax(total)))
        nce_loss /= -1. * batch_size * self.time_pred_step * self.pixel_pred_step
        acc = 1. * correct / batch_size

        return nce_loss.item(),acc.item()
    
    
    
if __name__ == "__main__":
    check = torch.randn(8,20,512,7,7)
    print(torch.randint(0,49,(1,)).long())
    cpc = twoD_CPC(time_pred_step = 2, pixel_pred_step = 2)
    out = cpc.forward(check)
    print(out)
    # print(out.shape)