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

class CrossAttention(nn.Module):
    """
    Compute the cross attention between the two modalities
    """
    def __init__(self, dim_embed, dim_project) -> None:
        super(CrossAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
        self.linear_key   = nn.Linear(dim_embed,dim_project)
        self.linear_query = nn.Linear(dim_embed,dim_project)
        self.linear_value = nn.Linear(dim_embed,dim_project)
    
    def forward(self,embed_1, embed_2):
        self.key_ = self.linear_key(embed_1)
        self.value_ = self.linear_value(embed_1)
        self.query_ = self.linear_query(embed_2)

        batch_size, length, d_tensor = embed_1.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = self.key_.transpose(1, 2)  # transpose
        score = (self.query_ @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ self.value_

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


class EncoderSmileLayer(nn.Module):
    def __init__(self, 
        d_model, n_head, drop_prob, expansion_factor, 
        bert_mask, lig_mask, 
        # complex_mask
    ):
        super(EncoderSmileLayer, self).__init__()
        self.bert_mask = bert_mask
        self.lig_mask = lig_mask
        # self.complex_mask = complex_mask

        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        #self.decode_smiles =  nn.Linear(767,d_model)

        self.ffn_bert = nn.Sequential(
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
        #x = self.decode_smiles(x)
        #print(x.shape)
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        
        # 3. positionwise feed forward network
        _x = x
        # x = self.ffn(x)
        # print("x", x.shape)
        # print("lig",self.lig_mask)
        # print("lig_shape", self.lig_mask.shape)
        # print("bert",self.bert_mask.shape)
        x_pro = self.ffn_bert(x[:,self.bert_mask])
        #print("x_pro", x_pro.shape)
        x_lig = self.ffn_lig(x[:, self.lig_mask])
        #print("lig", x_lig.shape)
        # print('FFN dim:', x.shape, x_pro.shape, x_lig.shape)
        # x_complex = self.ffn_complex(x[self.complex_mask])
        # x = torch.cat([x_pro, x_lig, x_complex], dim=1)
        x = torch.cat([x_pro, x_lig], dim=1)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

class EncoderProtLayer(nn.Module):
    def __init__(self, 
        d_model, n_head, drop_prob, expansion_factor, 
        pro_mask,
        # complex_mask
    ):
        super(EncoderProtLayer, self).__init__()
        self.pro_mask = pro_mask
        #self.lig_mask = lig_mask
        # self.complex_mask = complex_mask

        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn_pro = nn.Sequential(
            nn.Linear(d_model, expansion_factor*d_model),
            nn.ReLU(),
            nn.Linear(expansion_factor*d_model, d_model)
        )
        # self.ffn_lig = nn.Sequential(
        #     nn.Linear(d_model, expansion_factor*d_model),
        #     nn.ReLU(),
        #     nn.Linear(expansion_factor*d_model, d_model)
        # )
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
        #print("x", x.shape)
        #print("_x", _x.shape)
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        
        # 3. positionwise feed forward network
        _x = x
        # x = self.ffn(x)
        x_pro = self.ffn_pro(x)
        #x_lig = self.ffn_lig(x[:, self.lig_mask])
        # print('FFN dim:', x.shape, x_pro.shape, x_lig.shape)
        # x_complex = self.ffn_complex(x[self.complex_mask])
        # x = torch.cat([x_pro, x_lig, x_complex], dim=1)
        x = torch.cat([x_pro], dim=1)
        #print("x_1", x.shape)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

class ProLigBEiT(nn.Module):
    def __init__(self, 
        enc_voc_size, pro_len, lig_len,bert_len, d_model, n_head, n_layers, 
        expansion_factor, drop_prob, device
    ):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model=d_model, max_len=pro_len, device=device)
        self.tok_emb = nn.Embedding(num_embeddings=enc_voc_size, embedding_dim=d_model)
        self.pro_len = pro_len
        self.max_smiles_len = bert_len + lig_len
        pro_mask = torch.zeros(self.pro_len, dtype=torch.bool)
        lig_mask = torch.zeros(self.max_smiles_len, dtype=torch.bool)
        bert_mask = torch.zeros(self.max_smiles_len, dtype = torch.bool)
        bert_mask[:bert_len] = True
        lig_mask[bert_len:] = True
        self.cross_attention = CrossAttention(dim_embed=d_model,dim_project=d_model)
        self.linear_bert = nn.Linear(767, d_model)

        self.prot_layers = nn.ModuleList(
            [EncoderProtLayer(
                d_model=d_model, 
                n_head=n_head, 
                drop_prob=drop_prob,
                expansion_factor=expansion_factor,
                pro_mask=pro_mask,
                # complex_mask=complex_mask,
            )
            for _ in range(n_layers)]
        )

        self.smile_layers = nn.ModuleList(
            [EncoderSmileLayer(
                d_model=d_model, 
                n_head=n_head, 
                drop_prob=drop_prob,
                expansion_factor=expansion_factor,
                bert_mask=bert_mask,
                lig_mask=lig_mask
                # complex_mask=complex_mask,
            )
            for _ in range(n_layers)]
        )

        self.atten_head_out = nn.Sequential(
            nn.Linear(d_model,d_model//2),
            nn.Softplus(),
            nn.Linear(d_model,2)
        )
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, pro, lig_graph, lig_bert, src_mask=None):
        x_pro = self.tok_emb(pro) + self.pos_emb(pro)
        lig_bert = self.linear_bert(lig_bert)
        #print("lig",lig_bert.shape)
        x_lig = torch.cat([lig_graph, lig_bert], dim=1)
        #print("lig1",x_lig.shape)
        for layer in self.prot_layers:
            x_pro = layer(x_pro, src_mask)
        
        for layer_smi in self.smile_layers:
            x_lig = layer_smi(x_lig,src_mask)

        x_cross,_ = self.cross_attention(x_pro,x_lig)
        # print("x_c", x_cross.shape)

        # print(x_pro.shape)
        # print(x_lig.shape)

        x = x_cross[:, 0, :]
        x = self.pred_head(x)

        return x
