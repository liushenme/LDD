import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from torch.autograd import Variable
import math
from utils import Conv2d, Conv3d, Conv1d, ResConv3d
from torch.nn import Sequential, LeakyReLU, MaxPool3d, Module

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 501):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # make embeddings relatively larger
        # add constant to embedding
        
        seq_len = x.size(1)
        #print(seq_len)
        #print('pe', self.pe.shape)
        #assert seq_len<=self.max_seq_len
        if self.training:
            assert seq_len<=self.max_seq_len
            x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        else:
            if seq_len >= self.max_seq_len:
                pe = F.interpolate(
                    self.pe.transpose(1,2), seq_len, mode='linear', align_corners=False)
                #print('pe', self.pe.shape)
                #print('x', x.shape)
                x = x.transpose(1,2) + Variable(pe, requires_grad=False).cuda()
                x = x.transpose(1,2)
        return x

class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, positional_emb_flag, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.positional_emb_flag = positional_emb_flag
        if positional_emb_flag:
            self.positional_emb = PositionalEncoder(d_model)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        if self.positional_emb_flag:
            src = self.positional_emb(src)
            tar = self.positional_emb(tar)
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src

class attentionLayer_mask(nn.Module):

    def __init__(self, d_model, nhead, positional_emb_flag, dropout=0.1):
        super(attentionLayer_mask, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.positional_emb_flag = positional_emb_flag
        if positional_emb_flag:
            self.positional_emb = PositionalEncoder(d_model)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar, mask):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        mask = ~mask
        if self.positional_emb_flag:
            src = self.positional_emb(src)
            tar = self.positional_emb(tar)
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src

class Gate_2mlppooladd(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.g_c = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.g_hw = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, src, diff):

        g_hw = diff.permute(0,2,3,4,1)
        g_hw = self.g_hw(g_hw).permute(0,4,1,2,3)
        g_hw = F.adaptive_avg_pool3d(g_hw,(None,1, 1))
        out_hw = g_hw * src


        g_c = diff.permute(0,2,3,4,1)
        g_c = self.g_c(g_c)
        g_c = F.adaptive_avg_pool3d(g_c,(None,None,1)).permute(0,4,1,2,3)
        out_c = g_c * src

        out = out_hw + out_c

        return out

class Gate_2c3dpooladd(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.g_c = Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.g_hw = Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, src, diff):

        g_hw = self.g_hw(diff)
        g_hw = F.adaptive_avg_pool3d(F.sigmoid(g_hw),(None,1, 1))
        out_hw = g_hw * src

        g_c = self.g_c(diff)
        g_c = g_c.permute(0,2,3,4,1)
        g_c = F.adaptive_avg_pool3d(F.sigmoid(g_c),(None,None,1)).permute(0,4,1,2,3)
        out_c = g_c * src

        out = out_hw + out_c

        return out

class Gate_2c3dpoolcat(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.g_c = Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.g_hw = Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, src, diff):

        g_hw = self.g_hw(diff)
        g_hw = F.adaptive_avg_pool3d(F.sigmoid(g_hw),(None,1, 1))
        out_hw = g_hw * src

        g_c = self.g_c(diff)
        g_c = g_c.permute(0,2,3,4,1)
        g_c = F.adaptive_avg_pool3d(F.sigmoid(g_c),(None,None,1)).permute(0,4,1,2,3)
        out_c = g_c * src

        out = torch.cat((out_hw,out_c), 1)

        return out


class Gate_2c3dpoolhwc(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.g_c = Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.g_hw = Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, src, diff):

        g_hw = self.g_hw(diff)
        g_hw = F.adaptive_avg_pool3d(g_hw,(None,1, 1))
        out_hw = src + g_hw * src

        g_c = self.g_c(out_hw)
        g_c = g_c.permute(0,2,3,4,1)
        g_c = F.adaptive_avg_pool3d(g_c,(None,None,1)).permute(0,4,1,2,3)
        out_c = src + g_c * src

        return out_c

class Gate_2c3dpoolchw(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.g_c = Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.g_hw = Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, src, diff):

        g_c = self.g_c(diff)
        g_c = diff.permute(0,2,3,4,1)
        g_c = F.adaptive_avg_pool3d(g_c,(None,None,1)).permute(0,4,1,2,3)
        out_c = src + g_c * src

        g_hw = self.g_hw(out_c)
        g_hw = F.adaptive_avg_pool3d(g_hw,(None,1, 1))
        out_hw = src + g_hw * src
        return out_hw

class Gate_2c2dpoolhwc(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.g_c = Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.g_hw = Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, src, diff):

        g_hw = self.g_hw(diff)
        g_hw = F.sigmoid(F.adaptive_avg_pool2d(g_hw,(1, 1)))
        out_hw = g_hw * src

        g_c = self.g_c(diff)
        g_c = g_c.permute(0,2,3,1)
        g_c = F.sigmoid(F.adaptive_avg_pool2d(g_c,(None,1)).permute(0,3,1,2))
        out_c = g_c * out_hw

        return out_c

class Gate_2c2dpoolchw(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.g_c = Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.g_hw = Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, src, diff):

        g_c = self.g_c(diff)
        g_c = diff.permute(0,2,3,1)
        g_c = F.adaptive_avg_pool2d(g_c,(None,1)).permute(0,2,1,2)
        out_c = g_c * src

        g_hw = self.g_hw(diff)
        g_hw = F.adaptive_avg_pool2d(g_hw,(1, 1))
        out_hw = g_hw * out_c
        return out_hw

class Gate_2poolnnhwc(nn.Module):
    def __init__(self, in_dim, rate):
        super().__init__()
        self.g_c = nn.Sequential(
                    nn.Conv2d(1, rate, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(rate, 1, kernel_size=3, stride=1, padding=1)
                    )

        self.g_hw = nn.Sequential(
                    nn.Linear(in_dim, rate),
                    nn.ReLU(inplace=True), 
                    nn.Linear(rate, in_dim)
                    )

    def forward(self, src, diff):

        g_hw = F.adaptive_avg_pool2d(diff,(1, 1))
        g_hw = g_hw.permute(0,2,3,1)
        g_hw = F.sigmoid(self.g_hw(g_hw))
        g_hw = g_hw.permute(0,3,1,2)
        out_hw = g_hw * src

        g_c = diff.permute(0,2,3,1)
        g_c = F.adaptive_avg_pool2d(g_c,(None,1))
        g_c = F.sigmoid(self.g_c(g_c.permute(0,3,1,2)))
        out_c = g_c * out_hw

        return out_c

class Gate_2poolnnchw(nn.Module):
    def __init__(self, in_dim, rate):
        super().__init__()
        self.g_c = nn.Sequential(
                    nn.Conv2d(1, rate, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(rate, 1, kernel_size=3, stride=1, padding=1)
                    )

        self.g_hw = nn.Sequential(
                    nn.Linear(in_dim, rate),
                    nn.ReLU(inplace=True), 
                    nn.Linear(rate, in_dim)
                    )
    def forward(self, src, diff):
        g_c = diff.permute(0,2,3,1)
        g_c = F.adaptive_avg_pool2d(g_c,(None,1))
        g_c = F.sigmoid(self.g_c(g_c.permute(0,3,1,2)))
        out_c = g_c * src

        g_hw = F.adaptive_avg_pool2d(diff,(1, 1))
        g_hw = g_hw.permute(0,2,3,1)
        g_hw = F.sigmoid(self.g_hw(g_hw))
        g_hw = g_hw.permute(0,3,1,2)
        out_hw = g_hw * out_c

        return out_hw

class Gate_2poolnnchwa(nn.Module):
    def __init__(self, in_dim, rate):
        super().__init__()
        self.g_c = nn.Sequential(
                    nn.Conv2d(1, rate, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(rate, 1, kernel_size=3, stride=1, padding=1)
                    )

        self.g_hw = nn.Sequential(
                    nn.Linear(in_dim, rate),
                    nn.ReLU(inplace=True), 
                    nn.Linear(rate, in_dim)
                    )
    def forward(self, src, diff):
        g_c = diff.permute(0,2,3,1)
        g_c = F.adaptive_avg_pool2d(g_c,(None,1))
        g_c = self.g_c(g_c.permute(0,3,1,2))
        #out_c = g_c * src

        g_hw = F.adaptive_avg_pool2d(diff,(1, 1))
        g_hw = g_hw.permute(0,2,3,1)
        g_hw = self.g_hw(g_hw)
        g_hw = g_hw.permute(0,3,1,2)

        mask = F.sigmoid(g_c + g_hw)

        out_hw = mask * src

        return out_hw

class Gate_2poolnnchwres(nn.Module):
    def __init__(self, in_dim, rate):
        super().__init__()
        self.g_c = nn.Sequential(
                    nn.Conv2d(1, rate, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(rate, 1, kernel_size=3, stride=1, padding=1)
                    )

        self.g_hw = nn.Sequential(
                    nn.Linear(in_dim, rate),
                    nn.ReLU(inplace=True), 
                    nn.Linear(rate, in_dim)
                    )

    def forward(self, src, diff):
        g_c = diff.permute(0,2,3,1)
        g_c = F.adaptive_avg_pool2d(g_c,(None,1))
        g_c = F.sigmoid(self.g_c(g_c.permute(0,3,1,2)))
        out_c = src + g_c * src

        g_hw = F.adaptive_avg_pool2d(diff,(1, 1))
        g_hw = g_hw.permute(0,2,3,1)
        g_hw = F.sigmoid(self.g_hw(g_hw))
        g_hw = g_hw.permute(0,3,1,2)
        out_hw = out_c + g_hw * out_c

        return out_hw

class Gate_2poolnnhw(nn.Module):
    def __init__(self, in_dim, rate):
        super().__init__()
        self.g_hw = nn.Sequential(
                    nn.Linear(in_dim, rate),
                    nn.ReLU(inplace=True), 
                    nn.Linear(rate, in_dim)
                    )
    def forward(self, src, diff):

        g_hw = F.adaptive_avg_pool2d(diff,(1, 1))
        g_hw = g_hw.permute(0,2,3,1)
        g_hw = F.sigmoid(self.g_hw(g_hw))
        g_hw = g_hw.permute(0,3,1,2)
        out_hw = g_hw * src

        return out_hw

class Gate_2poolnnc(nn.Module):
    def __init__(self, in_dim, rate):
        super().__init__()
        self.g_c = nn.Sequential(
                    nn.Conv2d(1, rate, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(rate, 1, kernel_size=3, stride=1, padding=1)
                    )

    def forward(self, src, diff):
        g_c = diff.permute(0,2,3,1)
        g_c = F.adaptive_avg_pool2d(g_c,(None,1))
        g_c = F.sigmoid(self.g_c(g_c.permute(0,3,1,2)))
        out_c = g_c * src

        return out_c

