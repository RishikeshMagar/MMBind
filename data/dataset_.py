import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch.nn as nn

class ProtDataset(Dataset):
    def __init__(
        self, label_dir='label', ligand_dir='ligand_rep', seq_dir = 'seq_dir',
        smile_dir='sequence', max_smile_length = 227, max_prot_len = 261, train=True, num_embedding = 2,embedding_dim = 256
        ):
        # Max pocket seq length: 259, max # atoms in ligand: 224, max_length=259+224+2=485
        #max_length = max_length
        if train:
            self.ligand_dir = ligand_dir+'/other'
            self.smile_dir  = smile_dir#+'/other_pocket_seq.npy'
            self.label_dir  = label_dir+'/other.csv'
            seq_dir = seq_dir+'/other_pocket_seq.npy'
        else:
            self.ligand_dir = ligand_dir+'/refined'
            self.smile_dir       = smile_dir#+'/refined_pocket_seq.npy'
            self.label_dir       = label_dir+'/refined.csv'
            seq_dir = seq_dir+'/refined_pocket_seq.npy'
        """
        [CLS] Sequence+padding (max_seq_length-2) [SEP] Ligand+padding (224)
        """
        self.sep_tokens = {
            'CLS': 2,
            'PAD': 0,
            'SEP': 3
        }
        #print(self.label_dir)
        embedding = nn.Embedding(num_embedding, embedding_dim, max_norm=True)
        emb_tensor = torch.tensor([0,2,3])
        self.embedding = embedding(emb_tensor)
        #print(self.embedding.shape)
        self.labels_csv = pd.read_csv(self.label_dir, header=None,index_col=False).to_numpy()
        #print(self.labels_csv[0])
        self.labels = {self.labels_csv[i, 0].split('_')[0]: self.labels_csv[i, 1] for i in range(len(self.labels_csv))}
        print(len(self.labels))



        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        raw_seq = np.load(seq_dir, allow_pickle=True)
        print(len(raw_seq))
        self.prot_seq = []
        for i in range(len(raw_seq)):
            seq = raw_seq[i, 1].replace(' ', '')
            if len(seq) <= max_prot_len:
                tokens = self.tokenizer.encode(
                    raw_seq[i, 1], add_special_tokens=True, 
                    padding='max_length', max_length=max_prot_len
                )
                tokens[tokens.index(self.sep_tokens['SEP'])] = 0
                # tokens[-1] = self.sep_tokens['SEP']
                self.prot_seq.append([raw_seq[i, 0], tokens])
        self.prot_seq = np.array(self.prot_seq, dtype=object)
        self.prot_seq_dict = dict(self.prot_seq)
        #np.save("prot_seq.npy", self.prot_seq)


    
    """
    The length dataload is the number of complexes with 
    protein sequence length <= max_length - 224 (max n_atoms in ligand) 
    """
    def __len__(self):
        return len(self.labels_csv)

    """
    return of __getitem__:

    1. sequence with padding (numpy array, shape = [259+2]),
    2. equivariant node feature of ligand (torch Tensor, shape=[n_atoms, 256]),
    3. padding for the ligand node features (numpy array, shape = [224-n_atoms]),
    4. label of binding affinity
    """
    def __getitem__(self, idx):

        fname = self.labels_csv[idx][0]
        fname = fname.split('_')[0]
        ligand_path = os.path.join(self.ligand_dir, fname+'_ligand.pt')
        ligand_bert_path = os.path.join(self.smile_dir,fname+'.npy')
        ligand_bert_emb = torch.from_numpy(np.load(ligand_bert_path))
        #print(self.embedding)

        ligand_bert_cls = self.embedding[1]
        ligand_bert_sep = self.embedding[2]
        ligand_bert_pad = self.embedding[0]
        ligand_rep = torch.load(ligand_path)
        dim = ligand_rep.size() # dim = [n_atoms, 256]
        n_pad = 224 - dim[0]
        ligand_pad = torch.zeros(n_pad, 256)
        ligand_rep = torch.cat((ligand_rep, ligand_pad), dim=0)
        #print(ligand_rep.shape)
        #ligand_bert_emb = torch.transpose(ligand_bert_emb,0,1)
        #print("b",ligand_bert_cls.shape)
        ligand_bert_emb = torch.cat((ligand_bert_cls.unsqueeze(0),ligand_bert_emb,ligand_bert_sep.unsqueeze(0)), dim = 0)
        #print("a",ligand_bert_emb.shape)
        #print(ligand_rep.shape)


        pro_seq = torch.tensor(self.prot_seq_dict[fname], dtype=torch.long)
        label = torch.tensor(self.labels[fname], dtype=torch.float)
        return fname, pro_seq, ligand_rep, label, ligand_bert_emb