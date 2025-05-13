import os
import torch
import rasterio
import random
import numpy as np

#Սահմանումներ
data_root = "EuroSAT_MS_SCALED"
output_root = "EuroSAT_output_finallyga"

def read_multispectral_image(tif_path):
    with rasterio.open(tif_path) as src:
        img = src.read()
    return img

def save_tensor(tensor, split, cls, filename):
    out_path = os.path.join(output_root, split, cls, filename + ".pt")
    torch.save(tensor, out_path)

image_paths = []
class_names = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
for cls in class_names:
    class_dir = os.path.join(data_root, cls)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith('.tif'):
            image_paths.append((os.path.join(class_dir, fname), cls))
print(f"Found {len(image_paths)} images across {len(class_names)} classes.")

processed_band_indices = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
train_band_indices = [3, 4,6]
test_band_indices = [1,2,5,7,8,11,12]
print("Processed bands:", processed_band_indices)
print("Train bands:", train_band_indices)
print("Test bands:", test_band_indices)

# Տվյալների բաժանում ուսուցման, վալիդացիայի և թեստի
random.seed(42)
images_by_class = {cls: [] for cls in class_names}
for path, cls in image_paths:
    images_by_class[cls].append(path)

train_files, val_files, test_files = [], [], []
for cls, files in images_by_class.items():
    random.shuffle(files)
    n_total = len(files)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_files += [(f, cls) for f in files[:n_train]]
    val_files   += [(f, cls) for f in files[n_train:n_train + n_val]]
    test_files  += [(f, cls) for f in files[n_train + n_val:]]
    assert len(files[:n_train]) == n_train
    assert len(files[n_train:n_train + n_val]) == n_val
    assert len(files[n_train + n_val:]) == n_total - n_train - n_val
print(f"Train: {len(train_files)} images, Val: {len(val_files)} images, Test: {len(test_files)} images")

# Թղթապանակների ստեղծում
for split in ["train", "val", "test"]:
    for cls in class_names:
        os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

def augment_image(np_array):
    if random.random() < 0.5:
        np_array = np_array[:, :, ::-1]  # Հորիզոնական շրջում
    if random.random() < 0.5:
        np_array = np_array[:, ::-1, :]  # Ուղղահայաց շրջում
    k = random.randint(0, 3)
    if k:
        np_array = np.rot90(np_array, k=k, axes=(1, 2))  # Պտտում
    return np_array

# Ալիքների ընտրության ֆունկցիաներ
def select_channels_from_original(img, desired_indices):
    return np.stack([img[i] for i in desired_indices], axis=0)

def subset_channels(img, desired_orig_indices, processed_indices):
    idxs = [processed_indices.index(x) for x in desired_orig_indices]
    return img[idxs]

# Ուսուցման տվյալների մշակում
for filepath, cls in train_files:
    img = read_multispectral_image(filepath)
    img = select_channels_from_original(img, processed_band_indices)
    img = subset_channels(img, train_band_indices, processed_band_indices)
    img = augment_image(img)
    img_tensor = torch.from_numpy(img.copy()).float()
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    save_tensor(img_tensor, "train", cls, base_name)

# Վալիդացիայի տվյալների մշակում
for filepath, cls in val_files:
    img = read_multispectral_image(filepath)
    img = select_channels_from_original(img, processed_band_indices)
    img = subset_channels(img, train_band_indices, processed_band_indices)
    img_tensor = torch.from_numpy(img.copy()).float()
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    save_tensor(img_tensor, "val", cls, base_name)

# Թեստի տվյալների մշակում
for filepath, cls in test_files:
    img = read_multispectral_image(filepath)
    img = select_channels_from_original(img, processed_band_indices)
    img = subset_channels(img, test_band_indices, processed_band_indices)
    img_tensor = torch.from_numpy(img.copy()).float()
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    save_tensor(img_tensor, "test", cls, base_name)