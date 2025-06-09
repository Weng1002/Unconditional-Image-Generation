import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

from glob import glob
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from lpips import LPIPS  

# ======= 訓練設定 =======
image_size = 64
batch_size = 128
num_epochs = 800
learning_rate = 1e-5
timesteps = 1000
start_epoch = 1200
save_dir = "../model"
img_test = "../img_test"
data_dir = "../data"
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Positional Encoding =====
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)

# ===== Better UNet with Time =====
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.block(x)

class BetterUNetWithTime(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.bot = ConvBlock(base_ch * 4, base_ch * 8)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.dec3 = ConvBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2 + base_ch, base_ch)

        self.time_proj = nn.Linear(time_dim, base_ch * 8)
        self.final = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        temb = self.time_mlp(t)
        temb_proj = self.time_proj(temb).unsqueeze(-1).unsqueeze(-1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3)) + temb_proj

        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)

# ===== Diffusion Process =====
class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, timesteps=1000):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, timesteps))
        alphas = 1. - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))

    def forward(self, x_0):
        b, c, h, w = x_0.shape
        t = torch.randint(0, self.timesteps, (b,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * x_0 +
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        )
        pred_noise = self.model(x_t, t.float())
        loss_l1 = F.l1_loss(pred_noise, noise)
        loss_l2 = F.mse_loss(pred_noise, noise)
        return 0.25 * loss_l1 + 0.75 * loss_l2

    @torch.no_grad()
    def sample(self, batch_size):
        img = torch.randn(batch_size, 3, self.image_size, self.image_size).to(next(self.model.parameters()).device)
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t = torch.full((batch_size,), i, device=img.device, dtype=torch.float)
            noise_pred = self.model(img, t)
            beta = self.betas[i]
            alpha = 1. - beta
            alpha_cumprod = self.alphas_cumprod[i]

            noise = torch.randn_like(img) if i > 0 else 0
            img = (
                1 / torch.sqrt(alpha) * 
                (img - beta / torch.sqrt(1 - alpha_cumprod) * noise_pred) +
                torch.sqrt(beta) * noise
            )
        return img.clamp(0., 1.)
    
@torch.no_grad() 
def update_ema(model, ema_model, decay):
    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)

class RandomGaussianSharpen:
    def __call__(self, img):
        if random.random() < 0.3:
            img = TF.gaussian_blur(img, kernel_size=3)
        if random.random() < 0.3:
            img = TF.adjust_sharpness(img, sharpness_factor=2)
        return img

# ======= Dataset =======
class UnlabeledImageDataset(Dataset):
    def __init__(self, root, transform=None, extensions=("jpg", "jpeg", "png")):
        self.paths = []
        for ext in extensions:
            self.paths.extend(glob(os.path.join(root, f"*.{ext}")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Error loading {path}: {e}")
            image = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, 0

transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),  # 抗鋸齒插值
    transforms.CenterCrop(image_size),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)),  # 模糊去鋸齒（隨機程度）
    transforms.RandomApply([transforms.ColorJitter(0.05, 0.05, 0.05, 0.01)], p=0.3),  # 輕微增強
    transforms.ToTensor()
])

dataset = UnlabeledImageDataset(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# ======= 模型與 optimizer =======
model = BetterUNetWithTime(base_ch=64).to(device)
ema_model = deepcopy(model).to(device)
ema_decay = 0.9999
ema_start_step = 10000

model.load_state_dict(torch.load(f"{save_dir}/refine_model_epoch_1200.pt", map_location=device))
ema_model.load_state_dict(torch.load(f"{save_dir}/refine_model_1200_EMA.pt", map_location=device))

diffusion = GaussianDiffusion(model, image_size=image_size, timesteps=timesteps).to(device)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# ======= LPIPS Perceptual Loss =======
perceptual_loss_fn = LPIPS(net='vgg').to(device)

# ======= 訓練 =======
step = 0
for epoch in range(start_epoch, start_epoch + num_epochs):
    print(f"\n[Refine] Epoch {epoch+1}/{start_epoch + num_epochs}")
    pbar = tqdm(dataloader)
    running_loss = 0

    for imgs, _ in pbar:
        step += 1
        if step > ema_start_step:
            update_ema(model, ema_model, ema_decay)

        imgs = imgs.to(device)
        with torch.cuda.amp.autocast():
            b, c, h, w = imgs.shape
            t = torch.randint(0, timesteps, (b,), device=imgs.device).long()
            noise = torch.randn_like(imgs)
            x_t = (
                diffusion.sqrt_alphas_cumprod[t][:, None, None, None] * imgs +
                diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
            )
            pred_noise = model(x_t, t.float())
            recon = (
                (x_t - diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * pred_noise) /
                diffusion.sqrt_alphas_cumprod[t][:, None, None, None]
            )
            loss_l1 = F.l1_loss(pred_noise, noise)
            loss_l2 = F.mse_loss(pred_noise, noise)
            loss_perceptual = perceptual_loss_fn(recon, imgs).mean()
            loss = 0.25 * loss_l1 + 0.5 * loss_l2 + 0.25 * loss_perceptual

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_description(f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"[Refine] Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")


    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"{save_dir}/refine_model_epoch_{epoch+1}.pt")
        torch.save(ema_model.state_dict(), f"{save_dir}/refine_model_{epoch+1}_EMA.pt")
        with torch.no_grad():
            ema_diffusion = GaussianDiffusion(ema_model, image_size=image_size, timesteps=timesteps).to(device)
            sampled = ema_diffusion.sample(batch_size=16)
            save_image(sampled, f"{img_test}/refine_sample_epoch_{epoch+1:03d}.png", nrow=4)

# ======= 最終儲存 =======
torch.save(model.state_dict(), f"{save_dir}/final_refine_model.pt")
torch.save(ema_model.state_dict(), f"{save_dir}/final_refine_model_EMA.pt")
print("[Refine] 已儲存最終模型")