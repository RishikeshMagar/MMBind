import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import Dataset

class ProtDataset(Dataset):
    def __init__(
        self, label_dir='label', ligand_dir='ligand_rep', 
        seq_dir='sequence', max_length=485, train=True
        ):
        # Max pocket seq length: 259, max # atoms in ligand: 224, max_length=259+224+2=485
        max_seq_length = max_length - 224
        if train:
            self.ligand_dir = ligand_dir+'/other'
            seq_dir = seq_dir+'/other_pocket_seq.npy'
            label_dir = label_dir+'/other.csv'
        else:
            self.ligand_dir = ligand_dir+'/refined'
            seq_dir = seq_dir+'/refined_pocket_seq.npy'
            label_dir = label_dir+'/refined.csv'
        """
        [CLS] Sequence+padding (max_seq_length-2) [SEP] Ligand+padding (224)
        """
        self.sep_tokens = {
            'CLS': 2,
            'PAD': 0,
            'SEP': 3
        }
        self.labels = pd.read_csv(label_dir, header=None,index_col=False).to_numpy()
        self.labels = {self.labels[i, 0].split('_')[0]: self.labels[i, 1] for i in range(len(self.labels))}

        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        raw_seq = np.load(seq_dir, allow_pickle=True)
        self.prot_seq = []
        for i in range(len(raw_seq)):
            seq = raw_seq[i, 1].replace(' ', '')
            if len(seq) <= max_seq_length-2:
                tokens = self.tokenizer.encode(
                    raw_seq[i, 1], add_special_tokens=True, 
                    padding='max_length', max_length=max_seq_length
                )
                tokens[tokens.index(self.sep_tokens['SEP'])] = 0
                tokens[-1] = self.sep_tokens['SEP']
                self.prot_seq.append([raw_seq[i, 0], tokens])
        self.prot_seq = np.array(self.prot_seq, dtype=object)

    
    """
    The length dataload is the number of complexes with 
    protein sequence length <= max_length - 224 (max n_atoms in ligand) 
    """
    def __len__(self):
        return len(self.prot_seq)

    """
    return of __getitem__:

    1. sequence with padding (numpy array, shape = [259+2]),
    2. equivariant node feature of ligand (torch Tensor, shape=[n_atoms, 256]),
    3. padding for the ligand node features (numpy array, shape = [224-n_atoms]),
    4. label of binding affinity
    """
    def __getitem__(self, idx):
        fname = self.prot_seq[idx, 0]
        ligand_path = os.path.join(self.ligand_dir, fname+'_ligand.pt')
        ligand_rep = torch.load(ligand_path)
        dim = ligand_rep.size() # dim = [n_atoms, 256]
        n_pad = 224 - dim[0]
        ligand_pad = torch.zeros(n_pad, 256)
        ligand_rep = torch.cat((ligand_rep, ligand_pad), dim=0)

        pro_seq = torch.tensor(self.prot_seq[idx, 1], dtype=torch.long)
        label = torch.tensor(self.labels[fname], dtype=torch.float)
        return pro_seq, ligand_rep, label