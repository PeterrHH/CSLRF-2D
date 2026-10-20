import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn

audio_feat = torch.randn(32, 2048)
visual_appreance_feat = torch.randn(32, 8, 16, 2048)
visual_motion_feat = torch.randn(32, 8, 2048)
import math

class PositionalEncoding(nn.Module):



    def __init__(self, size: int = 0, max_len: int = 200):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]
class EncoderLayer(nn.Module):
    r"""EncoderLayer is mainly made up of self-attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pe = PositionalEncoding(d_model)
        self.activation = nn.ReLU()

    def forward(self, src):
        r"""Pass the input through the endocder layer.
        """
        src = src.permute(1, 0, 2)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.permute(1, 0, 2)
        return src
    
    
    



if __name__ == "__main__":
    enc = EncoderLayer(d_model = 512, nhead = 4, dim_feedforward = 1024, dropout = 0.1)
    ran = torch.rand(2,180,512)
    print(ran.shape)
    out = enc(ran)
    print(out.shape)
#     Visual_app_self_attention = EncoderLayer(d_model=2048, nhead=4, dim_feedforward=4, dropout=0.1)
#     Visual_mot_self_attention = EncoderLayer(d_model=2048, nhead=4, dim_feedforward=4, dropout=0.1)


#     if visual_motion_feat.size(1) != 2048:
#         print("false")
#     else:
#         print("true")

#     visual_motion_output = Visual_mot_self_attention(visual_motion_feat)
#     batch, num_clips, num_frames, _ = visual_appreance_feat.size()
#     visual_appreance_feat = visual_appreance_feat.reshape(batch, num_clips*num_frames, -1)
#     visual_appreance_output = Visual_app_self_attention(visual_appreance_feat)
#     visual_appreance_output = visual_appreance_feat.reshape(batch, num_clips, num_frames, -1)
#     print("visual_appreance_output:",visual_appreance_output.size())
