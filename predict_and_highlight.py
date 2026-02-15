import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# ======================
# Load trained model
# ======================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
).to(DEVICE)

model.load_state_dict(torch.load("nerve_model.pth", map_location=DEVICE))
model.eval()


# ======================
# Highlight function
# ======================
def highlight(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) / 255.0

    tensor = torch.tensor(resized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    overlay = img.copy()
    overlay[mask == 1] = [0, 255, 0]  # GREEN highlight

    cv2.imshow("Highlighted Nerve", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================
# Test on one image
# ======================
highlight(r"C:\Users\Admin\Desktop\nerve\train\1_1.tif")
