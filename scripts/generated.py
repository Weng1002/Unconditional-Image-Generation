import os
import math

import torch
import torch.nn as nn
from torchvision.utils import save_image

from tqdm import tqdm
import random

# ===== 基本參數設定 =====
image_size = 64
timesteps = 1000
num_images = 10000
batch_size = 256
   
model_ckpt = "../model/refine_model_1300_EMA.pt" 
save_dir = "../generated_images"

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

# ===== 載入模型與生成影像 =====
model = BetterUNetWithTime(base_ch=64).to(device)
diffusion = GaussianDiffusion(model, image_size=image_size, timesteps=timesteps).to(device)

print(f"載入模型中：{model_ckpt}")
model.load_state_dict(torch.load(model_ckpt, map_location=device))

# ===== 產生圖片 =====
n_generated = 0
pbar = tqdm(total=num_images, desc="Generating Images")

with torch.no_grad(), torch.cuda.amp.autocast():
    while n_generated < num_images:
        bsz = min(batch_size, num_images - n_generated)
        samples = diffusion.sample(batch_size=bsz)
        for i in range(samples.size(0)):
            save_path = os.path.join(save_dir, f"{n_generated:05d}.png")
            save_image(samples[i], save_path)
            n_generated += 1
            pbar.update(1)

print(f"已生成 {n_generated} 張圖片，儲存於：{save_dir}")
