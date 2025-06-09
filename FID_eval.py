import os
import numpy as np
import torch
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy import linalg
import argparse

def get_inception_model(device):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def extract_features(img_dir, model, transform, device, batch_size=32):
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if len(files) == 0:
        raise ValueError(f"No image files found in {img_dir}")

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(files), batch_size), desc="Extracting features"):
            batch = []
            for f in files[i:i + batch_size]:
                try:
                    img = Image.open(f).convert('RGB')
                    batch.append(transform(img))
                except Exception as e:
                    print(f"[WARNING] Failed to load image: {f} ({e})")
            if not batch:
                continue
            batch = torch.stack(batch).to(device)
            pred = model(batch)
            features.append(pred.cpu().numpy())

    if not features:
        raise ValueError("No valid features extracted. Check image files or transform.")
    
    return np.concatenate(features, axis=0)

def calculate_activation_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="image directory path")
    parser.add_argument("--ref_mu", type=str, default="./test_mu.npy", help="(e.g., test_mu.npy)")
    parser.add_argument("--ref_sigma", type=str, default="./test_sigma.npy", help="(e.g., test_sigma.npy)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using GPU: {device}")

    model = get_inception_model(device)
    transform = get_image_transform()

    features = extract_features(args.img_dir, model, transform, device=device)
    mu_fake, sigma_fake = calculate_activation_statistics(features)

    mu_real = np.load(args.ref_mu)
    sigma_real = np.load(args.ref_sigma)

    fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"[RESULT] FID score = {fid_score:.4f}")
    
    # python FID_eval.py --img_dir ./generated_images --ref_mu ./test_mu.npy --ref_sigma ./test_sigma.npy --gpu 0