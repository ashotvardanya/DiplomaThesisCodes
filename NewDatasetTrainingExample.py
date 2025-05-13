import os, random, numpy as np
from pathlib import Path
from tqdm import tqdm

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

ENABLE_RESIZE = True
IMG_SIZE      = 192
BATCH_SIZE    = 8
EPOCHS        = 5
LR            = 1e-4
NUM_WORKERS   = 4
SEED          = 42
DATA_DIR      = Path("experiment_3")
RUN_DIR       = Path("runs/experiment_3_sar_only")
RUN_DIR.mkdir(parents=True, exist_ok=True)

RGB_IDX = [10, 9, 8]
SAR_IDX = [3, 4]
TOTAL_BANDS = 18
BACKBONE = "so2sat_channelvit_small_p8_with_hcs_random_split_supervised"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PT18Dataset(Dataset):
    def __init__(self, root: Path, active_bands):
        self.paths, self.labels, self.classes = [], [], []
        self.active_bands = active_bands
        for cls in sorted(root.iterdir()):
            if not cls.is_dir(): continue
            lbl = len(self.classes)
            self.classes.append(cls.name)
            for pt in cls.glob("*.pt"):
                self.paths.append(pt); self.labels.append(lbl)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        x = torch.load(self.paths[idx])
        if ENABLE_RESIZE and x.shape[-1] != IMG_SIZE:
            x = F.interpolate(x.unsqueeze(0), IMG_SIZE,
                              mode='bilinear', align_corners=False).squeeze(0)
        mask = torch.zeros(TOTAL_BANDS, dtype=torch.long)
        mask[self.active_bands] = 1
        return x, self.labels[idx], mask

# Տվյալների բաժանումը (միայն SAR)
train_ds = PT18Dataset(DATA_DIR / "train" / "sar", SAR_IDX)
val_ds   = PT18Dataset(DATA_DIR / "val"   / "sar", SAR_IDX)
test_ds  = PT18Dataset(DATA_DIR / "test",            SAR_IDX)

n_classes = len(train_ds.classes)
print("▶ classes:", train_ds.classes,
      f"| train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True)
val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)
test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

model = torch.hub.load("insitro/ChannelViT",
                       BACKBONE, pretrained=True, trust_repo=True,
                       map_location=device, image_size=IMG_SIZE)
model.head = nn.Linear(model.num_features, n_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler    = GradScaler()
writer    = SummaryWriter(RUN_DIR)

# Ուսուցման/գնահատման ցիկլ
def run_epoch(loader, train: bool, desc: str):
    model.train() if train else model.eval()
    tot_loss = tot_ok = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb, m in tqdm(loader, desc=desc, ncols=80, leave=False):
            xb, yb, m = xb.to(device), yb.to(device), m.to(device)
            xb = xb * m.unsqueeze(-1).unsqueeze(-1)         # Ոչ ակտիվ բաժինները զրոյացնել
            with autocast():
                out = model(xb, extra_tokens={'channels': m})
                loss = criterion(out, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            tot_loss += loss.item()*xb.size(0)
            tot_ok   += (out.argmax(1)==yb).sum().item()
            n += xb.size(0)
    return tot_loss/n, 100.0*tot_ok/n

best = 0.0
for ep in range(1, EPOCHS+1):
    tl, ta = run_epoch(train_dl, True,  f"Train {ep}")
    vl, va = run_epoch(val_dl,   False, f"Val   {ep}")
    writer.add_scalars("Loss", {"train":tl, "val":vl}, ep)
    writer.add_scalars("Acc",  {"train":ta, "val":va}, ep)
    print(f"E{ep:02d}  Train {ta:5.2f}%  |  Val {va:5.2f}%")
    if va > best:
        best = va
        torch.save(model.state_dict(), RUN_DIR / "best_sar_only.pth")

# Թեստավորում
model.eval()
ts_loss, ts_acc = run_epoch(test_dl, False, "Test ")
print(f"★ Test accuracy (SAR → SAR): {ts_acc:5.2f}%")
writer.add_scalar("Acc/test", ts_acc); writer.close()