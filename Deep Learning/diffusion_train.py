import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    timesteps: (B,)
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, time_emb_dim=128):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.down2 = ResBlock(base_ch, base_ch * 2, time_emb_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.up1 = ResBlock(base_ch * 3, base_ch, time_emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.input_conv(x)

        d1 = self.down1(x, t_emb)           # [B, 32, 32, 32]
        d2 = self.down2(self.pool(d1), t_emb)  # [B, 64, 16, 16]

        m = self.mid(d2, t_emb)             # [B, 64, 16, 16]

        u = self.up(m)                      # [B, 64, 32, 32]
        u = torch.cat([u, d1], dim=1)       # [B, 96, 32, 32]
        u = self.up1(u, t_emb)              # [B, 32, 32, 32]

        return self.out_conv(F.silu(self.out_norm(u)))


class SimpleDiffusionModel(nn.Module):
    def __init__(self, T, device="cpu"):
        super().__init__()
        self.T = T
        self.device = device
        self.initialize_alpha_beta_schedules()

        self.unet = UNet(
            in_ch=3,
            out_ch=3,
            base_ch=32,
            time_emb_dim=128,
        ).to(device)

    def forward(self, x, t):
        return self.unet(x, t)  # predicted noise

    def loss(self, predicted_noise, actual_noise, writer=None, index=0):
        loss = F.mse_loss(predicted_noise, actual_noise)
        if writer is not None:
            writer.add_scalar("loss", loss.item(), index)
        return loss

    def initialize_alpha_beta_schedules(self):
        self.betas = torch.linspace(1e-4, 0.02, self.T).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)

        alpha_hat_t = self.alpha_hat[t]
        sqrt_alpha_hat = torch.sqrt(alpha_hat_t)[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1 - alpha_hat_t)[:, None, None, None]

        return sqrt_alpha_hat * x_0 + sqrt_one_minus * noise

    @torch.no_grad()
    def p_sample(self, x_t, t):
        one_minus_alpha_t = (1 - self.alphas[t])[:, None, None, None]
        sqrt_recip_alpha_t = (1.0 / torch.sqrt(self.alphas[t]))[:, None, None, None]
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        eps_theta = self.forward(x_t, t)

        x = sqrt_recip_alpha_t * (
            x_t - (one_minus_alpha_t / sqrt_one_minus_alpha_hat_t) * eps_theta
        )

        if t[0].item() == 0:
            return x
        else:
            noise = torch.randn_like(x_t)
            return x + torch.sqrt(self.betas[t[0]]) * noise


def sample_image(model, img_shape, device, n=8):
    model.eval()
    with torch.no_grad():
        x_t = torch.randn((n, *img_shape), device=device)
        for t in reversed(range(model.T)):
            t_batch = torch.tensor([t] * n, device=device)
            x_t = model.p_sample(x_t, t_batch)

    return x_t


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    # training_data = datasets.MNIST(root='./data_src', train=True, download=True,transform=transform)
    # test_data = datasets.MNIST(root='./data_src', train=False, download=True, transform=transform)

    training_data = datasets.CIFAR10(root='./data_src', train=True, download=True,transform=transform)
    test_data = datasets.CIFAR10(root='./data_src', train=False, download=True, transform=transform)
    IMG_SHAPE = (3, 32, 32)
    training_dataloader = DataLoader(training_data,
                                    batch_size=64,
                                    shuffle=True,
                                    drop_last=True)
    test_dataloader = DataLoader(test_data,
                                batch_size=64,
                                shuffle=False,
                                drop_last=True)
    test_loader_iter = iter(test_dataloader)

    T = 200 # number of diffusion steps
    model = SimpleDiffusionModel(T, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_writer = SummaryWriter(log_dir='runs/cifar/train')
    test_writer = SummaryWriter(log_dir='runs/cifar/test')

    step = 0
    EPOCHS = 200
    for epoch in range(EPOCHS):
        for index, (x0, _) in enumerate(training_dataloader):
            model.train()
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.shape[0],), device=device)
            actual_noise = torch.randn_like(x0, device=device)
            xt = model.q_sample(x0, t, actual_noise)  # noisy image
            predicted_noise = model(xt, t)
            loss = model.loss(predicted_noise, actual_noise, writer=train_writer, index=step)

            if index % 100 == 0:
                with torch.no_grad():
                    try:
                        tbatch = next(test_loader_iter)
                    except StopIteration:
                        test_loader_iter = iter(test_dataloader)
                        tbatch = next(test_loader_iter)
                    x0_test = tbatch[0].to(device)
                    t_test = torch.randint(0, T, (x0_test.shape[0],), device=device)
                    actual_noise_test = torch.randn_like(x0_test, device=device)
                    xt_test = model.q_sample(x0_test, t_test, actual_noise_test)  # noisy image
                    predicted_noise_test = model(xt_test, t_test)
                    test_loss = model.loss(predicted_noise_test, actual_noise_test, writer=test_writer, index=step)
                    images = sample_image(model, IMG_SHAPE, device, n=8)
                    grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
                    test_writer.add_image('test_images', grid, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        print(f'Epoch [{epoch+1}/{EPOCHS}] Step [{index+1}/{len(training_dataloader)}] Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'simple_diffusion_model.pth')
