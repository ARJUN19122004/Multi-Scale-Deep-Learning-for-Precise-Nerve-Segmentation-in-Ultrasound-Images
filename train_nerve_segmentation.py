import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_FOLDER = r"C:\Users\SEC\Downloads\nerve_segmentation-main\nerve_segmentation-main\train"
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30   # ✅ increased to 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
class NerveDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.images = [
            f for f in os.listdir(folder)
            if f.endswith(".tif") and "_mask" not in f
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".tif", "_mask.tif")

        img_path = os.path.join(self.folder, img_name)
        mask_path = os.path.join(self.folder, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image or mask: {img_name}")

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0

        image = torch.tensor(image).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask

# =========================
# MAIN (Windows Safe)
# =========================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # ✅ Windows fix

    dataset = NerveDataset(DATA_FOLDER)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    print(f"Total training images: {len(dataset)}")

    # =========================
    # MODEL
    # =========================
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    ).to(DEVICE)

    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = torch.sigmoid(model(imgs))  # ✅ correct
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), "nerve_model.pth")
    print("✅ Training complete. Model saved as nerve_model.pth")

