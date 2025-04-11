import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from torchvision import transforms

# === Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODEL_PATH = "swin_chestxdet10_best.pth"  # Your trained Swin Transformer path
TEST_IMG_DIR = "dataset/test_data/test_data"  # Test images

# === Categories
category_to_index = {
    'Consolidation': 0,
    'Pneumothorax': 1,
    'Emphysema': 2,
    'Calcification': 3,
    'Nodule': 4,
    'Mass': 5,
    'Fracture': 6,
    'Effusion': 7,
    'Atelectasis': 8,
    'Fibrosis': 9,
    'No Finding': 10
}
index_to_category = {v: k for k, v in category_to_index.items()}

# === Load model
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(category_to_index))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Grad-CAM++ for Swin

def generate_swin_gradcampp(model, img_path, target_class):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    activations = {}
    gradients = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook

    # Hook target layer
    target_layer = model.layers[-1].blocks[-1].norm1
    handle_fwd = target_layer.register_forward_hook(save_activation("feat"))
    handle_bwd = target_layer.register_backward_hook(save_gradient("feat"))

    # Forward + backward
    output = model(input_tensor)
    score = output[0, target_class]
    model.zero_grad()
    score.backward()

    act = activations["feat"].squeeze(0)  # [49, C]
    grad = gradients["feat"].squeeze(0)   # [49, C]

    if act.ndim == 3:
        act = act.reshape(-1, act.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1])

    alpha_num = grad.pow(2)
    alpha_denom = 2 * grad.pow(2) + (act * grad.pow(3)).sum(dim=0, keepdim=True)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alpha = alpha_num / (alpha_denom + 1e-7)
    weights = (alpha * grad).sum(dim=0)

    cam = torch.matmul(act, weights)
    cam = cam.reshape(7, 7).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # Overlay
    img_np = np.array(img.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap / 255.0 + img_np
    overlay = overlay / overlay.max()

    handle_fwd.remove()
    handle_bwd.remove()

    return (np.uint8(255 * overlay), float(torch.sigmoid(output[0, target_class]).item()))

# === Visualize multiple test images

def visualize_gradcam_grid(class_index=0, num_images=8):
    class_name = index_to_category[class_index]
    test_images = sorted(os.listdir(TEST_IMG_DIR))

    selected_imgs = []
    for img_name in test_images:
        if len(selected_imgs) >= num_images:
            break
        if img_name.endswith(".png") or img_name.endswith(".jpg"):
            selected_imgs.append(os.path.join(TEST_IMG_DIR, img_name))

    plt.figure(figsize=(16, 6))
    for idx, img_path in enumerate(selected_imgs):
        overlay, prob = generate_swin_gradcampp(model, img_path, target_class=class_index)
        plt.subplot(2, num_images // 2, idx + 1)
        plt.imshow(overlay)
        plt.title(f"{class_name} ({prob:.2f})")
        plt.axis("off")

    plt.suptitle(f"Grad-CAM++ on ChestXDet10 (Swin Transformer)\nClass: {class_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

# === Run it
visualize_gradcam_grid(class_index=10, num_images=8)  # Example: Consolidation class