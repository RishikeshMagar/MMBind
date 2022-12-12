import os
import numpy as np
from datetime import datetime
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from  sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from data.dataset_ import ProtDataset
from model.cross_transformer import ProLigBEiT
from lr_sched import adjust_learning_rate


seed = 0

device = 'cuda'
n_epochs = 10
valid_size = 0.2
batch_size = 64
num_workers = 0

lr = 1e-3
min_lr = 1e-7
warmup_epochs = 10
patience_epochs = 0
weight_decay = 1e-5

d_model = 256
n_head = 8
n_layers = 4
expansion_factor = 2
drop_prob = 0.25

train_dataset = ProtDataset(
    label_dir='./data/label', ligand_dir='./data/ligand_rep', seq_dir= './data/sequence',
    smile_dir='./data/embedding_other', train=True,
    num_embedding = 4,embedding_dim = 767
)
valid_dataset = ProtDataset(
    label_dir='./data/label', ligand_dir='./data/ligand_rep', seq_dir = './data/sequence',
    smile_dir='./data/embedding_refined', train=False,
    num_embedding = 4,embedding_dim = 767
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, drop_last=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, drop_last=True
)

model = ProLigBEiT(
    enc_voc_size=30, pro_len=261, lig_len=224, bert_len = 3, d_model=d_model, n_head=n_head, 
    n_layers=n_layers, expansion_factor=expansion_factor, drop_prob=drop_prob, device=device
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

save_dir = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join('ckpt', save_dir)
os.makedirs(save_dir, exist_ok=True)
best_valid_loss = float('inf')

for e in range(n_epochs):
    for bn, (__, prot, lig, label,bert_emb) in enumerate(train_loader):
        curr_epoch = e + bn / len(train_loader)
        adjust_learning_rate(optimizer, curr_epoch, lr, min_lr, n_epochs, warmup_epochs, patience_epochs)
        prot, lig, label, bert_emb = prot.to(device), lig.to(device), label.to(device), bert_emb.to(device)
        out = model(prot, lig,bert_emb)
        loss = F.mse_loss(out.squeeze(), label)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if bn % 50 == 0:
            print(f'Epoch: {e+1}/{n_epochs}, Batch: {bn}, Loss: {loss.item()}')

    # Validation
    valid_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        model.eval()
        for __, prot, lig, label, bert_emb in valid_loader:
            prot, lig, label, bert_emb = prot.to(device), lig.to(device), label.to(device), bert_emb.to(device)
            out = model(prot, lig, bert_emb)
            loss = F.mse_loss(out.squeeze(), label)
            valid_loss += loss.item()
            preds.append(out.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        rmse = mean_squared_error(labels, preds, squared=False)
        mae = mean_absolute_error(labels, preds)
        pearson = pearsonr(labels, preds)[0][0]
        print(f'Epoch: {e+1}, Valid Loss: {valid_loss/len(valid_loader):.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R: {pearson:.4f}')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
        np.save("preds.npy", preds)
        np.save("labels.npy", labels)