from typing import Dict, Optional, Union, Sequence, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.ffv_pt import Metadata
from .blocks import Contraction2, Contraction_small, Contraction_mid, Conv1d_block, MultiLayerProjectionNet

from .attentionLayer import *
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from .segmentation_2 import VisionLanguageBlock
from .video_encoder import C3DVideoEncoder, C3DVideoEncoder_small, C3DVideoEncoder_flow_small, C3DVideoEncoder_small_4d, C3DVideoEncoder_flow_small_4d, CNNEncoder_small
from .audioEncoder import seaudioEncoder
from .visualEncoder import visualtalenet 

from utils import Conv3d, Conv1d, ResConv3d
from einops import rearrange
import torchvision.models as models

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.                                                                                      
    
    See description of make_non_pad_mask.                                                                                                       
    
    Args:
        lengths (torch.Tensor): Batch of lengths (B,).                                                                                          
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.                                                                            
    
    Examples:                  
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask

class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''          
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear = nn.Linear(d_input, output_size)
        
    def forward(self, x, n_wins):
        #print(x.shape[1])       
        #mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = torch.arange(x.shape[1]).unsqueeze(0) < n_wins.unsqueeze(1).to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), n_wins.unsqueeze(1))   
            
        x = self.linear(x)
        
        return x

class PoolAvg_2(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''          
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear = nn.Linear(d_input, output_size)
        
    def forward(self, x, n_wins):
        x = self.linear(x)
        x = torch.mean(x, dim = 1)
        return x



class Deepfakecla_rgb_c2d_difffusion_4d_audiofusion(LightningModule):
    def __init__(self,
        v_encoder: str = "talk", a_encoder: str = "se", frame_classifier: str = "lr",
        ve_features=(64, 96, 128, 128), ae_features=(32, 64, 64), v_cla_feature_in=256, a_cla_feature_in=256,
        boundary_features=(512, 128), boundary_samples=10, temporal_dim=512, max_duration=40,
        weight_frame_loss=2., weight_modal_bm_loss=1., weight_contrastive_loss=0.1, contrast_loss_margin=0.99,
        weight_decay=0.0001, learning_rate=0.0002, distributed=False
    ):
        super().__init__()
        #self.save_hyperparameters()

        if v_encoder == "res":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.video_encoder = nn.Sequential(*list(resnet.children())[:-2]) 
            for param in self.video_encoder.parameters():
                param.requires_grad = False

            self.video_proj = nn.Conv2d(512, 128, kernel_size=1, stride=1)
            self.flow_proj = nn.Conv2d(512, 128, kernel_size=1, stride=1)

        elif v_encoder == "talk":
            self.video_encoder = visualtalenet()
        elif v_encoder == "light":
            self.video_encoder = visual_encoder()
        elif v_encoder == "pre":
            self.video_encoder = Conv1d_block(4098, 256)
            #self.flow_encoder = Contraction_small(2048, 256, 128, 0)
            #self.video_encoder = Res1dNet12()
            
        if a_encoder == "cnn":
            self.audio_encoder = CNNAudioEncoder(n_features=ae_features)
        elif a_encoder == "se":
            self.audio_encoder = seaudioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
        elif a_encoder == "pre":
            self.audio_encoder = nn.Conv1d(768, 128, kernel_size=1, stride=1)


        self.ca = VisionLanguageBlock(d_model=128, nhead=4, dim_feedforward=512, dropout=0)

        self.frame_loss = CrossEntropyLoss()
            
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed
        self.pool = PoolAvg(128, 2)

        
        self.gate = Gate_2poolnnchw(128, 16)


    def forward(self, video: Tensor, audio: Tensor, n_frames):
        # encoders

        mask = make_pad_mask(n_frames)
        mask = mask.unsqueeze(1).bool()
        
        b, c, t, h, w = video.shape
        video_r = rearrange(video, "b c t h w -> (b t) c h w")

        v_features_o = self.video_encoder(video_r)
        v_features = self.video_proj(v_features_o)
        #v_features_t1 = self.video_proj(v_features_o)
        v_features_t1 = v_features
        #print(v_features.shape)

        v_features = rearrange(v_features, "(b t) c h w -> b c t h w", b=b, t=t)
        v_features_t1 = rearrange(v_features_t1, "(b t) c h w -> b c t h w", b=b, t=t)

        diff = v_features[:, :, 1:, :, :] - v_features_t1[:, :, :-1, :, :]
        diff = F.pad(diff, (0, 0, 0, 0, 0, 1, 0, 0), mode="constant", value=0) 

        b, c, t, h, w = v_features.shape
        v_features = rearrange(v_features, "b c t h w -> (b t) c h w").contiguous()
        diff = rearrange(diff, "b c t h w -> (b t) c h w").contiguous()
        fu_features = self.gate(v_features, diff)
        fu_features = rearrange(fu_features, "(b t) c h w -> b c t h w", b=b, t=t).contiguous()

        fu_features = F.adaptive_avg_pool2d(fu_features, 1).squeeze(-1).squeeze(-1).transpose(1,2)

        #audio = audio.transpose(1,2)
        #a_features, _ = self.audio_encoder(audio, mask)
        a_features = self.audio_encoder(audio).transpose(1,2)
        #fu_features_2 = torch.cat((fu_features, a_features), 2)
        mask_bool = ~mask.squeeze(1)
        #print(mask_bool)
        pos = None
        fu_features_2 = self.ca(tgt=fu_features,
                             memory=a_features,
                             tgt_key_padding_mask=mask_bool,
                             memory_key_padding_mask=mask_bool,
                             pos=pos,
                             query_pos=pos) # [t*h*w, b, c]

        fu_cla = self.pool(fu_features_2, n_frames)
        
        return fu_cla 

    def loss_fn(self, fu_label: Tensor, fu_cla) -> Dict[str, Tensor]:

        #fu_frame_loss = self.frame_loss(fu_cla.squeeze(1), fu_label.float())
        fu_frame_loss = self.frame_loss(fu_cla, fu_label)

        loss = fu_frame_loss

        return {
            "loss": loss 
        }

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:
        video, audio, n_frames, fu_label, _ = batch

        fu_cla = self(video, audio, n_frames)
        loss_dict = self.loss_fn(fu_label, fu_cla)

        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        video, audio, n_frames, fu_label, _ = batch
        #print(fu_label)
        fu_cla = self(video, audio, n_frames)
        loss_dict = self.loss_fn(fu_label, fu_cla)

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }

    @staticmethod
    def get_meta_attr(meta: Metadata, video: Tensor, audio: Tensor):

        fu_label = 1.0 if meta.modify_audio or meta.modify_video else 0.0
        
        return [meta.video_frames, fu_label]

