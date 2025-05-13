import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import rasterio
from tqdm.auto import tqdm

DATA_DIR = "/kaggle/input/eurosatms/EuroSAT_MS_SCALED"
LOGDIR = "runs/ChannelViT_EuroSAT_bandGreedy_full"
os.makedirs(LOGDIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B11", "B12"]
IDX_10 = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
NUM_CLS = 10
LR = 1e-4
EPOCHS = 5
BATCH = 32
torch.manual_seed(0);
np.random.seed(0);
random.seed(0)
writer = SummaryWriter(LOGDIR)


def read_img(path):
    with rasterio.open(path) as src:
        arr = src.read()[IDX_10].astype(np.float32)
    return (arr / 1e4 if arr.max() > 255 else arr).clip(0, 1)


# Տվյալների բաժանումներ
classes = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d)))
cls2idx = {c: i for i, c in enumerate(classes)}
all_files = [(os.path.join(DATA_DIR, c, f), cls2idx[c])
             for c in classes
             for f in os.listdir(os.path.join(DATA_DIR, c)) if f.endswith(".tif")]
random.shuffle(all_files)
n = len(all_files)
s1, s2 = int(0.8 * n), int(0.9 * n)
train_f, val_f, test_f = all_files[:s1], all_files[s1:s2], all_files[s2:]
print(f"Dataset: train={len(train_f)} val={len(val_f)} test={len(test_f)}")


class EuroSAT10(Dataset):
    MEAN, STD = None, None

    def __init__(self, files, mask=None, aug=False):
        self.files, self.mask, self.aug = files, mask, aug
        if EuroSAT10.MEAN is None:
            samp = np.stack([read_img(p) for p, _ in random.sample(files, 64)])
            EuroSAT10.MEAN = samp.mean((0, 2, 3))
            EuroSAT10.STD = samp.std((0, 2, 3)) + 1e-6

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p, y = self.files[idx]
        x = read_img(p)
        if self.mask is not None:
            x = x * self.mask[:, None, None]
        if self.aug and random.random() < 0.5: x = x[:, :, ::-1]
        if self.aug and random.random() < 0.5: x = x[:, ::-1, :]
        if self.aug:
            k = random.randint(0, 3)
            if k: x = np.rot90(x, k, (1, 2))
        x = torch.from_numpy(np.ascontiguousarray(x))
        x = (x - torch.from_numpy(EuroSAT10.MEAN[:, None, None])) \
            / torch.from_numpy(EuroSAT10.STD[:, None, None])
        return x, y


# ChannelViT մոդելի բեռնում
def load_model():
    m = torch.hub.load(
        'insitro/ChannelViT',
        'so2sat_channelvit_small_p8_with_hcs_random_split_supervised',
        pretrained=True, trust_repo=True
    )
    proj = m.patch_embed.proj
    if isinstance(proj, nn.Conv3d) and proj.in_channels != 1:
        proj.weight = nn.Parameter(proj.weight.mean(1, keepdim=True))
        proj.in_channels = 1
    dim = getattr(m, 'num_features', None) or getattr(m, 'embed_dim', None)
    if dim is None: dim = m.norm.normalized_shape[0]
    m.head = nn.Linear(dim, NUM_CLS)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    return m.to(DEVICE)


CHAN_TOK = torch.arange(10, device=DEVICE).unsqueeze(0)


def forward_net(net, x):
    return net(x, extra_tokens={'channels': CHAN_TOK.repeat(x.size(0), 1)})


# Տրված mask-ի գնահատում
def eval_mask(mask, desc=""):
    tr_loader = DataLoader(EuroSAT10(train_f, mask, aug=True), BATCH, True, num_workers=2)
    val_loader = DataLoader(EuroSAT10(val_f, mask, aug=False), BATCH, False, num_workers=2)
    ts_loader = DataLoader(EuroSAT10(test_f, 1 - mask, aug=False), BATCH, False, num_workers=2)
    net = load_model()
    opt = optim.AdamW(net.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        # Ուսուցում
        net.train()
        train_pbar = tqdm(tr_loader, desc=f"{desc} ► epoch {epoch}/{EPOCHS}", leave=False)
        for x, y in train_pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = ce(forward_net(net, x), y)
            loss.backward();
            opt.step()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        # Վալիդացիա
        net.eval()
        tot = ok = 0
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            ok += (forward_net(net, x).argmax(1) == y).sum().item()
            tot += y.size(0)
        val_acc = ok / tot
        writer.add_scalar(f"{desc}/val_acc", val_acc, epoch)

    # Լրացուցիչ ալիքի գոտու թեստ
    net.eval()
    tot = ok = 0
    for x, y in ts_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        ok += (forward_net(net, x).argmax(1) == y).sum().item()
        tot += y.size(0)
    return ok / tot


# Ագահ առաջընթաց ընտրություն
current_mask = np.zeros(10, dtype=np.float32)
best_comp_acc = 0.0
step = 0
print("\n>>> Starting greedy search from empty mask <<<\n")

while True:
    best_candidate = None
    best_candidate_acc = best_comp_acc
    for b in range(10):
        if current_mask[b] == 0:
            test_mask = current_mask.copy();
            test_mask[b] = 1
            print(f"▶ Trying mask: {test_mask.astype(int).tolist()} …")
            acc = eval_mask(test_mask, f"step{step}_add_{BANDS[b]}")
            print(f"    → comp-band test acc: {acc * 100:.2f}%\n")
            if acc > best_candidate_acc:
                best_candidate_acc = acc
                best_candidate = b
    if best_candidate is None:
        break
    current_mask[best_candidate] = 1
    best_comp_acc = best_candidate_acc
    print(f"*** Added {BANDS[best_candidate]} → mask={current_mask.astype(int).tolist()}, "
          f"comp-acc={best_comp_acc * 100:.2f}%\n")
    step += 1
#վերջնական արդյունքներ
print("\n==== FINAL SELECTED MASK ====")
print("Train bands:", [BANDS[i] for i, v in enumerate(current_mask) if v])
print("Test  bands:", [BANDS[i] for i, v in enumerate(current_mask) if not v])
print(f"Complementary-band accuracy: {best_comp_acc * 100:.2f}%")
writer.close()
print(f"TensorBoard logs in {LOGDIR}")