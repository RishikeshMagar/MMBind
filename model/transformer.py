import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re
import pandas as pd
import numpy as np


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class EncoderLayer(nn.Module):
    def __init__(self, 
        d_model, n_head, drop_prob, expansion_factor, 
        pro_mask, lig_mask, 
        # complex_mask
    ):
        super(EncoderLayer, self).__init__()
        self.pro_mask = pro_mask
        self.lig_mask = lig_mask
        # self.complex_mask = complex_mask

        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # self.ffn = nn.Sequential(
        #     nn.Linear(d_model, expansion_factor*d_model),
        #     nn.ReLU(),
        #     nn.Linear(expansion_factor*d_model, d_model)
        # )
        self.ffn_pro = nn.Sequential(
            nn.Linear(d_model, expansion_factor*d_model),
            nn.ReLU(),
            nn.Linear(expansion_factor*d_model, d_model)
        )
        self.ffn_lig = nn.Sequential(
            nn.Linear(d_model, expansion_factor*d_model),
            nn.ReLU(),
            nn.Linear(expansion_factor*d_model, d_model)
        )
        # self.ffn_complex = nn.Sequential(
        #     nn.Linear(d_model, expansion_factor*d_model),
        #     nn.ReLU(),
        #     nn.Linear(expansion_factor*d_model, d_model)
        # )

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask=None):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        
        # 3. positionwise feed forward network
        _x = x
        # x = self.ffn(x)
        x_pro = self.ffn_pro(x[:, self.pro_mask])
        x_lig = self.ffn_lig(x[:, self.lig_mask])
        x = torch.cat([x_pro, x_lig], dim=1)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x


class ProLigBEiT(nn.Module):
    def __init__(self, 
        enc_voc_size, pro_len, lig_len, d_model, n_head, n_layers, 
        expansion_factor, drop_prob, device
    ):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model=d_model, max_len=pro_len, device=device)
        self.tok_emb = nn.Embedding(num_embeddings=enc_voc_size, embedding_dim=d_model)
        self.pro_modality_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)
        self.lig_modality_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.max_len = pro_len + lig_len
        pro_mask = torch.zeros(self.max_len, dtype=torch.bool)
        lig_mask = torch.zeros(self.max_len, dtype=torch.bool)
        pro_mask[:pro_len] = True
        lig_mask[pro_len:] = True

        self.layers = nn.ModuleList(
            [EncoderLayer(
                d_model=d_model, 
                n_head=n_head, 
                drop_prob=drop_prob,
                expansion_factor=expansion_factor,
                pro_mask=pro_mask,
                lig_mask=lig_mask,
                # complex_mask=complex_mask,
            )
            for _ in range(n_layers)]
        )

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, pro, lig, src_mask=None):
        pro = self.tok_emb(pro) + self.pos_emb(pro)

        # pro_mod = torch.zeros(pro.size(0), pro.size(1), dtype=torch.long).to(pro.device)
        # lig_mod = torch.ones(lig.size(0), lig.size(1), dtype=torch.long).to(pro.device)
        # pro = pro + self.pro_modality_emb(pro_mod)
        # lig = lig + self.lig_modality_emb(lig_mod)

        x = torch.cat([pro, lig], dim=1)

        for layer in self.layers:
            x = layer(x, src_mask)

        x = x[:, 0, :]
        x = self.pred_head(x)

        return x
