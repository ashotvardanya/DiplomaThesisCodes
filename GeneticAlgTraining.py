import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data      import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import rasterio, tqdm.auto as tq

DATA_DIR = "/kaggle/input/eurosatms/EuroSAT_MS_SCALED"
LOGDIR   = "runs/ChannelViT_EuroSAT_bandGA"
os.makedirs(LOGDIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BANDS   = ["B02","B03","B04","B05","B06","B07","B08","B08A","B11","B12"]
IDX_10  = [1,2,3,4,5,6,7,8,11,12]
NUM_CLS = 10

LR       = 1e-4
EPOCHS   = 5
BATCH    = 32
POP      = 6
GEN      = 5
MUT_P    = 0.4
TRAIN_SIZES = {3,4,5}
RGB_IDX  = []

torch.manual_seed(0); np.random.seed(0); random.seed(0)
writer = SummaryWriter(LOGDIR)


def read_img(path:str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()[IDX_10].astype(np.float32)
    return (arr/1e4 if arr.max()>255 else arr).clip(0,1)

# Տվյալների ֆայլերի ցուցակների կառուցում
classes = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR,d)))
cls2idx = {c:i for i,c in enumerate(classes)}
files = [(os.path.join(DATA_DIR,c,f), cls2idx[c])
         for c in classes
         for f in os.listdir(os.path.join(DATA_DIR,c)) if f.endswith(".tif")]
random.shuffle(files)
n = len(files); s1, s2 = int(0.8*n), int(0.9*n)
train_f, val_f, test_f = files[:s1], files[s1:s2], files[s2:]
print(f"• dataset split  train/val/test = {len(train_f)}/{len(val_f)}/{len(test_f)}")

class EuroSAT10(Dataset):
    MEAN, STD = None, None
    def __init__(self, file_list, band_mask=None, aug=False):
        self.f = file_list; self.mask = band_mask; self.aug = aug
        if EuroSAT10.MEAN is None:
            samp = np.stack([read_img(p) for p,_ in random.sample(file_list,64)])
            EuroSAT10.MEAN = samp.mean((0,2,3)); EuroSAT10.STD = samp.std((0,2,3))+1e-6
    def __len__(self): return len(self.f)
    def _rand_aug(self,x):
        if random.random()<0.5: x = x[:,:,::-1]
        if random.random()<0.5: x = x[:, ::-1,:]
        k = random.randint(0,3); return np.rot90(x,k,(1,2)) if k else x
    def __getitem__(self, idx):
        p,lab = self.f[idx]; x = read_img(p)
        if self.mask is not None: x *= self.mask[:,None,None]
        if self.aug: x = self._rand_aug(x)
        x = torch.from_numpy(np.ascontiguousarray(x))
        x = (x - torch.from_numpy(EuroSAT10.MEAN[:,None,None])) / \
                torch.from_numpy(EuroSAT10.STD[:,None,None])
        return x, lab


# ChannelViT մոդելի բեռնում
def load_channelvit():
    m = torch.hub.load(
        'insitro/ChannelViT',
        'so2sat_channelvit_small_p8_with_hcs_random_split_supervised',
        pretrained=True, trust_repo=True
    )
    proj = m.patch_embed.proj
    if isinstance(proj, nn.Conv3d) and proj.in_channels != 1:
        proj.weight = nn.Parameter(proj.weight.mean(1, keepdim=True))
        proj.in_channels = 1
        print("∙ patched Conv3d input channels → 1")
    dim = getattr(m,'num_features',None) or getattr(m,'embed_dim',None)
    if dim is None: dim = m.norm.normalized_shape[0]
    m.head = nn.Linear(dim, NUM_CLS)
    if torch.cuda.device_count() > 1: m = nn.DataParallel(m)
    return m.to(DEVICE)

CHAN_TOKEN = torch.tensor([list(range(10))], dtype=torch.long, device=DEVICE)
def fwd(net,x):
    return net(x, extra_tokens={'channels': CHAN_TOKEN.repeat(x.size(0),1)})


# Գենետիկ ալգորիթմի օգնական ֆունկցիաներ
def valid_mask(m):
    return m.sum() in TRAIN_SIZES and all(m[i] for i in RGB_IDX)

def rand_mask():
    while True:
        m = np.zeros(10, dtype=np.float32)
        k = random.choice(list(TRAIN_SIZES))
        m[random.sample(range(10),k)] = 1
        if valid_mask(m): return m

def mutate(m):
    for _ in range(10):
        c = m.copy()
        i1 = random.choice(np.where(c==1)[0])
        i0 = random.choice(np.where(c==0)[0])
        c[i1] = 0; c[i0] = 1
        if valid_mask(c): return c
    return m

def xover(a,b):
    for _ in range(10):
        pt = random.randint(1,9)
        child = np.concatenate([a[:pt], b[pt:]])
        if valid_mask(child): return child
    return a

def base_masks():
    vis=[0,1,2]; rededge=[3,4,5]; nir=[6]; swir=[8,9]; narrow=[7]
    m37 = np.zeros(10); m37[vis+nir]          = 1
    m46 = np.zeros(10); m46[vis+rededge[:1]+nir] = 1
    m55 = np.zeros(10); m55[vis+rededge[:2]]  = 1
    return [m.astype(np.float32) for m in (m37,m46,m55)]


# Ալիքի գոտու վրա ուսուցում
def run_split(mask, gen_id, ind_id):
    train_ds = EuroSAT10(train_f, mask, aug=True)
    val_ds   = EuroSAT10(val_f  , mask, aug=False)
    tr = DataLoader(train_ds, BATCH, True , num_workers=2, pin_memory=True)
    va = DataLoader(val_ds  , BATCH, False, num_workers=2, pin_memory=True)

    net = load_channelvit()
    opt = optim.AdamW(net.parameters(), lr=LR)
    ce  = nn.CrossEntropyLoss()

    pbar = tq.tqdm(range(1,EPOCHS+1), desc=f"gen{gen_id}‑ind{ind_id}")
    for ep in pbar:
        net.train(); loss_sum=0
        for x,y in tr:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = ce(fwd(net,x), y); loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0)
        loss_mean = loss_sum/len(train_ds)
        net.eval(); ok=tot=0
        with torch.no_grad():
            for x,y in va:
                x,y = x.to(DEVICE),y.to(DEVICE)
                ok  += (fwd(net,x).argmax(1)==y).sum().item(); tot+=y.size(0)
        val_acc = ok/tot
        pbar.set_postfix(loss=f"{loss_mean:.4f}", val=f"{val_acc*100:.2f}%")
        step = (gen_id*POP + ind_id)*EPOCHS + ep
        writer.add_scalar("train/loss", loss_mean, step)
        writer.add_scalar("val/acc",   val_acc,   step)

    # Մյուս ալիքների վրա թեստ
    tst_ds = EuroSAT10(test_f, 1-mask, aug=False)
    ts = DataLoader(tst_ds, BATCH, False, num_workers=2, pin_memory=True)
    net.eval(); ok=tot=0
    with torch.no_grad():
        for x,y in ts:
            x,y = x.to(DEVICE),y.to(DEVICE)
            ok += (fwd(net,x).argmax(1)==y).sum().item(); tot+=y.size(0)
    return ok/tot


# Գենետիկ ալգորիթմի գլխավոր ցիկլ
fitness = {}
pop = base_masks()
while len(pop) < POP:
    pop.append(rand_mask())

for g in range(GEN):
    print(f"\n========= Generation {g} =========")
    for i,m in enumerate(pop):
        key = tuple(m)
        if key not in fitness:
            print("mask", m.astype(int).tolist())
            acc = run_split(m, g, i)
            fitness[key] = acc
            writer.add_scalar("GA/fitness", acc, g*POP + i)
            print(f"  → complementary‑band acc  {acc*100:.2f}%")
    pop.sort(key=lambda m: fitness[tuple(m)], reverse=True)
    elite1, elite2 = pop[:2]
    print("best so far:", fitness[tuple(elite1)]*100)
    new = [elite1, elite2]
    while len(new) < POP:
        child = xover(elite1, elite2)
        if random.random()<MUT_P: child = mutate(child)
        new.append(child)
    pop = new
#Վերջնական լավագույնը
best = max(fitness, key=fitness.get)
print("\n==== FINAL BEST MASK ====")
print("train bands :", [BANDS[i] for i,b in enumerate(best) if b])
print("test  bands :", [BANDS[i] for i,b in enumerate(best) if not b])
print("val‑acc on unseen bands :", f"{fitness[best]*100:.2f}%")

writer.close()
print(f"TensorBoard log saved to {LOGDIR}")