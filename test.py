import os
import csv
import numpy as np
from datetime import datetime
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from  sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

from data.dataset import ProtDataset
from model.transformer import ProLigBEiT


seed = 0

device = 'cuda'
valid_size = 0.2
batch_size = 64
num_workers = 4

d_model = 256
n_head = 8
n_layers = 4
expansion_factor = 2
drop_prob = 0

# dataset = ProtDataset(
#     label_dir='data/id_prop.csv', 
#     ligand_dir='data/ligand_rep', 
#     # seq_dir='data/prot_seq.npy'
#     seq_dir='data/pocket_seq.npy'
# )

# num_data = len(dataset)
# indices = list(range(num_data))
# # np.random.shuffle(indices)
# random_state = np.random.RandomState(seed=seed)
# random_state.shuffle(indices)

# split = int(np.floor(valid_size * num_data))
# valid_idx, train_idx = indices[:split], indices[split:]
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)
# train_loader = DataLoader(
#     dataset, batch_size=batch_size, sampler=train_sampler,
#     num_workers=num_workers, drop_last=True
# )
# valid_loader = DataLoader(
#     dataset, batch_size=batch_size, sampler=valid_sampler,
#     num_workers=num_workers, drop_last=True
# )

train_dataset = ProtDataset(
    label_dir='data/label', ligand_dir='data/ligand_rep', 
    seq_dir='data/sequence', train=True
)
valid_dataset = ProtDataset(
    label_dir='data/label', ligand_dir='data/ligand_rep', 
    seq_dir='data/sequence', train=False
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
    enc_voc_size=30, pro_len=261, lig_len=224, d_model=d_model, n_head=n_head, 
    n_layers=n_layers, expansion_factor=expansion_factor, drop_prob=drop_prob, device=device
).to(device)
# model_dir = 'ckpt/Nov30_13-28-01'
# model_dir = 'ckpt/Nov30_13-37-21'
# model_dir = 'ckpt/Dec04_18-22-35' # w\ modality embedding
# model_dir = 'ckpt/Dec04_19-24-51'
model_dir = 'ckpt/Dec04_20-04-41' # w\ modality embedding, w\ FFN
model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), map_location=device))

ids, preds, labels = [], [], []
with torch.no_grad():
    model.eval()
    for fname, prot, lig, label in valid_loader:
        prot, lig, label = prot.to(device), lig.to(device), label.to(device)
        out = model(prot, lig)
        loss = F.mse_loss(out.squeeze(), label)
        ids.extend(fname)
        preds.append(out.squeeze().detach().cpu().numpy())
        labels.append(label.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    rmse = mean_squared_error(labels, preds, squared=False)
    mae = mean_absolute_error(labels, preds)
    pearson = pearsonr(labels, preds)[0]
    r2 = r2_score(labels, preds)

# test and save predictions
with open(os.path.join(model_dir, 'result.csv'), mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(labels)):
        csv_writer.writerow([ids[i], labels[i], preds[i]])
    csv_writer.writerow(['RMSE', rmse])
    csv_writer.writerow(['MAE', mae])
    csv_writer.writerow(['R', pearson])
    csv_writer.writerow(['R2', r2])
    print('RMSE: {:.4f}, MAE: {:.4f}, R: {:.4f}, R2: {:.4f}'.format(rmse, mae, pearson, r2))

plt.figure(figsize=(7,7), dpi=600)
plt.scatter(labels, preds, c='b', marker='+')
p1 = max(max(preds), max(labels)) + 0.5
p2 = min(min(preds), min(labels)) - 0.5
plt.plot([p1, p2], [p1, p2], 'k--')
plt.xlabel('Labels', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
# plt.show()
plt.savefig(os.path.join(model_dir, 'plot.png'), bbox_inches='tight')