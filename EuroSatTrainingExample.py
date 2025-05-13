#!/usr/bin/env python3

import math
import os
import random
import torch
import numpy as np
from torch import nn, optim, hub
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Կարգավորումներ
DATA_DIR   = "/kaggle/input/eurosat-256/EuroSAT_output_finally"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 32
EPOCHS     = 10
SEED       = 42
USE_DINO   = False
LOG_DIR    = "runs/eurosat_experiment"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Տվյալների հավաքածուի ստեղծում
class EuroSATPTDataset(Dataset):
    def __init__(self, root, channel_map, aug=False, drop=False, p=0.1):
        self.channel_map, self.aug, self.drop, self.p = channel_map, aug, drop, p
        self.samples, self.class_to_idx = [], {}
        for cls in sorted(os.listdir(root)):
            if cls.startswith('.'): continue
            self.class_to_idx.setdefault(cls, len(self.class_to_idx))
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(cls_dir, f),
                                         self.class_to_idx[cls]))
        self.n_cls = len(self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, label = self.samples[idx]
        x = torch.load(fp)
        _, H, W = x.shape

        full = torch.zeros(18, H, W)
        for i, pos in enumerate(self.channel_map):
            full[pos] = x[i]

        mask = torch.zeros(18, dtype=torch.long)
        mask[self.channel_map] = 1

        if self.drop:
            keep = (torch.rand(len(self.channel_map)) > self.p).float()
            if keep.sum() == 0:
                keep[random.randrange(len(self.channel_map))] = 1.0
            for i, pos in enumerate(self.channel_map):
                full[pos] *= keep[i]
            mask[self.channel_map] = keep.long()

        if self.aug:
            if random.random() < 0.5: full = torch.flip(full, dims=[2])
            if random.random() < 0.5: full = torch.flip(full, dims=[1])
            k = random.randint(0, 3)
            if k: full = torch.rot90(full, k, dims=(1, 2))

        return full, label, mask

# Մոդելների բեռնիչներ
def load_so2sat_supervised(num_classes, device="cuda"):
    model = torch.hub.load(
        'insitro/ChannelViT',
        'so2sat_channelvit_small_p8_with_hcs_random_split_supervised',
        pretrained=True,
        map_location=device
    )
    model.head = nn.Linear(model.num_features, num_classes)
    return model

def load_dino_selfsupervised(num_classes, in_chans=18, device="cuda"):
    model = hub.load(
        'insitro/ChannelViT',
        'imagenet_channelvit_small_p16_DINO',
        pretrained=True,
        map_location=device
    )
    # Conv3d կշռի ընդլայնում
    conv = model.patch_embed.proj
    w    = conv.weight
    reps = [1]*w.dim()
    reps[1] = math.ceil(in_chans / w.shape[1])
    w_big = w.repeat(*reps)[:, :in_chans]
    conv.weight = nn.Parameter(w_big.contiguous())
    conv.in_channels = in_chans

    # channel_embed-ի ընդլայնում
    ce   = model.patch_embed.channel_embed
    reps = [1]*ce.dim()
    reps[2] = math.ceil(in_chans / ce.shape[2])
    ce_big = ce.repeat(*reps)[:, :, :in_chans]
    model.patch_embed.channel_embed = nn.Parameter(ce_big.contiguous())

    model.head = nn.Linear(model.num_features, num_classes)
    return model

# Տվյալների հավաքածուների և բեռնիչների պատրաստում
train_ds = EuroSATPTDataset(
    os.path.join(DATA_DIR, "train"),
    channel_map=[8,12,13],
    aug=True, drop=True
)
val_ds = EuroSATPTDataset(
    os.path.join(DATA_DIR, "val"),
    channel_map=[8,12,13]
)
test_ds = EuroSATPTDataset(
    os.path.join(DATA_DIR, "test"),
    channel_map=[9,10,11,14,15,16,17]
)

train_dl = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=4, pin_memory=True)
test_dl  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=4, pin_memory=True)

# Մոդելի և օպտիմիզատորի կառուցում
if USE_DINO:
    model = load_dino_selfsupervised(train_ds.n_cls, in_chans=18, device=DEVICE)
else:
    model = load_so2sat_supervised(train_ds.n_cls, device=DEVICE)

model = model.to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

base       = model.module if isinstance(model, nn.DataParallel) else model
head_params= list(base.head.parameters())
head_set   = set(head_params)
back_params= [p for p in model.parameters() if p.requires_grad and p not in head_set]

optimizer  = optim.AdamW([
    {"params": back_params, "lr": 3e-5},
    {"params": head_params, "lr": 3e-4}
])
criterion  = nn.CrossEntropyLoss()
scaler     = GradScaler()

# TensorBoard 
writer = SummaryWriter(log_dir=LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

# Ուսուցում և վալիդացիա
for epoch in range(1, EPOCHS+1):
    # Ուսուցում
    model.train()
    tr_loss = tr_corr = tr_tot = 0
    for x, y, ch in tqdm(train_dl, desc=f"Train {epoch}/{EPOCHS}"):
        x, y, ch = x.to(DEVICE), y.to(DEVICE), ch.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            out  = model(x, extra_tokens={"channels": ch})
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss   += loss.item() * x.size(0)
        tr_corr   += (out.argmax(1)==y).sum().item()
        tr_tot    += x.size(0)
    train_acc = 100 * tr_corr / tr_tot
    train_loss= tr_loss / tr_tot
    print(f"Epoch {epoch} Train  Acc: {train_acc:.2f}%  Loss: {train_loss:.3f}")

    writer.add_scalar("Train/Accuracy", train_acc, epoch)
    writer.add_scalar("Train/Loss",     train_loss, epoch)

    # Վալիդացիա
    model.eval()
    v_loss = v_corr = v_tot = 0
    with torch.no_grad(), autocast():
        for x, y, ch in val_dl:
            x, y, ch = x.to(DEVICE), y.to(DEVICE), ch.to(DEVICE)
            out  = model(x, extra_tokens={"channels": ch})
            loss = criterion(out, y)
            v_loss   += loss.item() * x.size(0)
            v_corr   += (out.argmax(1)==y).sum().item()
            v_tot    += x.size(0)
    val_acc  = 100 * v_corr / v_tot
    val_loss = v_loss / v_tot
    print(f"         Val    Acc: {val_acc:.2f}%  Loss: {val_loss:.3f}")

    writer.add_scalar("Val/Accuracy", val_acc, epoch)
    writer.add_scalar("Val/Loss",     val_loss, epoch)

# Թեստավորում և վերջնական գրանցում
model.eval()
te_corr = te_tot = 0
with torch.no_grad(), autocast():
    for x, y, ch in tqdm(test_dl, desc="Test"):
        x, y, ch = x.to(DEVICE), y.to(DEVICE), ch.to(DEVICE)
        out = model(x, extra_tokens={"channels": ch})
        te_corr += (out.argmax(1)==y).sum().item()
        te_tot  += x.size(0)
test_acc = 100 * te_corr / te_tot
print(f"\n✔ TEST ACCURACY {test_acc:.2f}%")

writer.add_scalar("Test/Accuracy", test_acc, EPOCHS)
writer.close()