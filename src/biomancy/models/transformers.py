import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import copy
from typing import Optional


class OmicsTransformer(nn.Module):
    '''
    Omics Transformer model that solves classification task
    '''
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        max_seq_len: int,
        feature_dim: int,
        d_model: Optional[int] = 512,
        nhead: Optional[int] = 8,
        num_encoder_layers: Optional[int] = 6,
        dim_feedforward: Optional[int] = 2048,
        dropout: Optional[float] = 0.1,
        activation:  Optional[str] = 'relu',
        ) -> None:
        '''
        Args:
            num_classes (int): Number of classes
            vocab_size (int):  Vocabulary size Defines the different k-mers that
                can be passed as input to the forward method
            max_seq_len (int): The maximum sequence length
            feature_dim (int): Dimension of feature vector
            d_model (Optional[int]): Dimension of Encoder Layer
            nhead (Optional[int]): Number of attention heads
            num_encoder_layers (Optional[int]): Number of Encoder layers
            dim_feedforward (Optional[int]): Dimension of Feedforward block
            dropout (Optional[float]): Dropout probabilitiy
            activation (Optional[str]): Type of activation function
        '''
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.class_embed = nn.Linear(d_model, num_classes)
        self.feature_embed = nn.Linear(feature_dim, d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.d_model = d_model
        self.nhead = nhead
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class OmicsTransformerСlassification(OmicsTransformer):
    '''
    Omics Transformer model that solves classification task
    '''
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        max_seq_len: int,
        feature_dim: int,
        d_model: Optional[int] = 512,
        nhead: Optional[int] = 8,
        num_encoder_layers: Optional[int] = 6,
        dim_feedforward: Optional[int] = 2048,
        dropout: Optional[float] = 0.1,
        activation:  Optional[str] = 'relu',
        device: Optional[torch.device] = None,
        ) -> None:
        '''
        Args:
            num_classes (int): Number of classes
            vocab_size (int):  Vocabulary size of the model. Defines the different k-mers that
                can be passed as input to the forward method
            max_seq_len (int): The maximum sequence length
            feature_dim (int): Dimension of feature vector
            d_model (Optional[int]): Dimension of Encoder Layer
            nhead (Optional[int]): Number of attention heads
            num_encoder_layers (Optional[int]): Number of Encoder layers
            dim_feedforward (Optional[int]): Dimension of Feedforward block
            dropout (Optional[float]): Dropout probabilitiy
            activation (Optional[str]): Type of activation function
            device (Optional[torch.device]): Object representing the device on which a torch.Tensor is or will be allocated
        '''
        super().__init__(num_classes, vocab_size, max_seq_len, feature_dim, d_model, nhead, num_encoder_layers,
                            dim_feedforward, dropout, activation)
      
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pe = positionalencoding1d(d_model, max_seq_len + 1, device)
        self.to_latent = nn.Identity()
        self._init_parameters()

    def forward(
        self,
        src: torch.IntTensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        '''
        Args:
            src (torch.IntTensor): Tensor of shape (batch, max_seq_len)
                Input tokens
            features (torch.Tensor): Tensor of shape (batch, max_seq_len, feature_dim)
                Omics data
            mask (Optional[torch.Tensor]): Tensor of shape (batch, max_seq_len)
                Mask indicating which elements within key to ignore for the purpose of attention (treat as padding)

        Returns:
            Tensor: Tensor of shape (batch, num_classes)
                Classification scores (before SoftMax)
        '''
        src = self.embed(src)
        src += self.feature_embed(features)
        b, _, _ = src.shape
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        src = torch.cat((cls_tokens, src), dim=1) 
        mask = torch.cat((torch.tensor([[False]]*mask.shape[0], device=mask.get_device()), mask), dim=1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=self.pe)
        
        x = self.to_latent(memory[:,0])
        outputs_class = self.class_embed(x)  
        return outputs_class


class OmicsTransformerSegmentation(OmicsTransformer):
    '''
    Omics Transformer model that solves segmentation task
    '''
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        max_seq_len: int,
        feature_dim: int,
        d_model: Optional[int] = 512,
        nhead: Optional[int] = 8,
        num_encoder_layers: Optional[int] = 6,
        dim_feedforward: Optional[int] = 2048,
        dropout: Optional[float] = 0.1,
        activation:  Optional[str] = 'relu',
        device: Optional[torch.device] = None,
        ) -> None:
        '''
        Args:
            num_classes (int): Number of classes
            vocab_size (int):  Vocabulary size of the model. Defines the different k-mers that
                can be passed as input to the forward method
            max_seq_len (int): The maximum sequence length
            feature_dim (int): Dimension of feature vector
            d_model (Optional[int]): Dimension of Encoder Layer
            nhead (Optional[int]): Number of attention heads
            num_encoder_layers (Optional[int]): Number of Encoder layers
            dim_feedforward (Optional[int]): Dimension of Feedforward block
            dropout (Optional[float]): Dropout probabilitiy
            activation (Optional[str]): Type of activation function
            device (Optional[torch.device]): Object representing the device on which a torch.Tensor is or will be allocated
        '''
        super().__init__(num_classes, vocab_size, max_seq_len, feature_dim, d_model, nhead, num_encoder_layers,
                            dim_feedforward, dropout, activation)

        self.pe = positionalencoding1d(d_model, max_seq_len, device)
        self._init_parameters()

    def forward(
        self,
        src: torch.IntTensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        '''
        Args:
            src (torch.IntTensor): Tensor of shape (batch, max_seq_len)
                Input tokens
            features (torch.Tensor): Tensor of shape (batch, max_seq_len, feature_dim)
                Omics data
            mask (Optional[torch.Tensor]): Tensor of shape (batch, max_seq_len)
                Mask indicating which elements within key to ignore for the purpose of attention (treat as padding)

        Returns:
            Tensor: Tensor of shape (batch, max_seq_len, num_classes)
                Classification scores (before SoftMax)
        '''
        src = self.embed(src)
        src += self.feature_embed(features)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=self.pe)
        outputs_class = self.class_embed(memory)
        return outputs_class


class TransformerEncoderLayer(nn.Module):
    '''
    Transformer Encoder Layer block
    '''
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = 2048,
        dropout: Optional[float] = 0.1,
        activation: Optional[str] = 'relu',
        ) -> None:
        '''
        Args:
            d_model (int): Dimension of Encoder Layer
            nhead (int): Number of attention heads
            dim_feedforward (Optional[int]): Dimension of Feedforward block
            dropout (Optional[float]): Dropout probabilitiy
            activation (Optional[str]): Type of activation function
        '''
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None, 
        pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        '''
        Args:
            src (torch.Tensor): Tensor of shape (batch, max_seq_len, d_model)
            src_key_padding_mask (Optional[torch.Tensor]): Tensor of shape (batch, max_seq_len)
                Mask indicating which elements within key to ignore for the purpose of attention (treat as padding)
            pos (Optional[torch.Tensor]): Tensor of shape (max_seq_len, d_model)
                Position embedding

        Returns:
            Tensor: Tensor of shape (batch, max_seq_len, d_model)
        '''
        # add position embedding in Encoder Layer according to DETR (https://arxiv.org/abs/2005.12872)
        q = k = self._add_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    def _add_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        '''
        Add position embedding to input tensor
        '''
        return tensor if pos is None else tensor + pos


class TransformerEncoder(nn.Module):
    '''
    Transformer Encoder block
    '''
    def __init__(
        self, 
        encoder_layer: TransformerEncoderLayer, 
        num_layers: int,
        ) -> None:
        '''
        Args:
            encoder_layer (TransformerEncoderLayer): Object of TransformerEncoderLayer
            num_layers (int): Number of Encoder layers
        '''
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self, 
        src: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None, 
        pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        '''
        Args:
            src (torch.Tensor): Tensor of shape (batch, max_seq_len, d_model)
            src_key_padding_mask (Optional[torch.Tensor]): Tensor of shape (batch, max_seq_len)
                Mask indicating which elements within key to ignore for the purpose of attention (treat as padding)
            pos (Optional[torch.Tensor]): Tensor of shape (max_seq_len, d_model)
                Position embedding

        Returns:
            Tensor: Tensor of shape (batch, max_seq_len, d_model)
        '''
        for layer in self.layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return src


def positionalencoding1d(
    d_model: int, 
    length: int, 
    device: torch.device,
    ) -> torch.Tensor:
    '''
    This version of the position embedding is described in the Attention is all you need 
    (https://arxiv.org/abs/1706.03762)

    Args:
        d_model (int): Dimension of Encoder Layer
        length (int): The maximum number of input tokens
        device (torch.device): Object representing the device on which a torch.Tensor is or will be allocated

    Returns:
        Tensor: Tensor of shape (length, d_model)
            Position embedding
    '''
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def _get_activation_fn(activation: str):
    '''
    Return an activation function given a string
    '''
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')

def _get_clones(module: nn.Module(), N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])