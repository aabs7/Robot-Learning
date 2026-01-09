import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image


def log_images(writer, images, labels, nrow, step):
    grid_shape = (nrow, math.ceil(images.shape[0] / nrow))
    fig_size = (grid_shape[1] * 2, grid_shape[0] * 2)
    fig = plt.figure(figsize=fig_size, dpi=200)
    is_grayscale = images.shape[1] == 1

    for i in range(images.shape[0]):
        ax = fig.add_subplot(grid_shape[0], grid_shape[1], i + 1)
        img_np = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
        ax.imshow(img_np, cmap='gray' if is_grayscale else None)
        ax.axis('off')
        ax.set_title(labels[i].item())
    with io.BytesIO() as buff:
        plt.savefig(buff, format='png', bbox_inches='tight')
        buff.seek(0)
        img = Image.open(buff)
        img = img.convert('L') if is_grayscale else img.convert('RGB')
        img = np.array(img)
        img = img[:, :, None] if is_grayscale else img

    writer.add_image('test_images', img, step, dataformats='HWC')
    plt.close(fig)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


class TimeClassEmbedding(nn.Module):
    def __init__(self, time_dim, num_classes):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, t, y):
        '''
        t: (B, ) tensor of timesteps
        y: (B, ) tensor of class labels
        '''
        t_emb = self.time_mlp(t)
        y_emb = self.class_emb(y)
        return t_emb + y_emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.activation = nn.SiLU()

    def forward(self, x, cond_emb):
        h = self.activation(self.conv1(x))
        h = h + self.cond_proj(cond_emb)[:, :, None, None]
        h = self.activation(self.conv2(h))
        return h


class UnetDown(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        '''Downscale the image features'''
        self.conv = ConvBlock(in_ch, out_ch, cond_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, cond_emb):
        x = self.conv(x, cond_emb)
        skip = x
        x = self.pool(x)
        return x, skip


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        '''Upscale the image features'''
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch=out_ch*2, out_ch=out_ch, cond_dim=cond_dim)

    def forward(self, x, skip, cond_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1) # concatenate along channel dimension
        x = self.conv(x, cond_emb)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, base_channels=128, num_classes=10, cond_dim=256):
        super().__init__()

        self.cond_embedding = TimeClassEmbedding(cond_dim, num_classes)

        # encoder
        self.down1 = UnetDown(in_ch, base_channels, cond_dim)
        self.down2 = UnetDown(base_channels, base_channels * 2, cond_dim)
        self.down3 = UnetDown(base_channels * 2, base_channels * 4, cond_dim)

        # bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, cond_dim)

        # decoder
        self.up3 = UnetUp(base_channels * 8, base_channels * 4, cond_dim)
        self.up2 = UnetUp(base_channels * 4, base_channels * 2, cond_dim)
        self.up1 = UnetUp(base_channels * 2, base_channels, cond_dim)

        # output
        self.out_conv = nn.Conv2d(base_channels, in_ch, kernel_size=1)


    def forward(self, x, t, y):
        '''
        x: (B, C, H, W) noisy image
        t: (B, ) timesteps
        y: (B, ) class labels
        '''

        cond_embed = self.cond_embedding(t, y)
        x, s1 = self.down1(x, cond_embed)
        x, s2 = self.down2(x, cond_embed)
        x, s3 = self.down3(x, cond_embed)
        x = self.bottleneck(x, cond_embed)

        x = self.up3(x, s3, cond_embed)
        x = self.up2(x, s2, cond_embed)
        x = self.up1(x, s1, cond_embed)

        return self.out_conv(x)


class SimpleDiffusionModel(nn.Module):
    def __init__(self, T, num_channels=3, num_classes=10, device="cpu"):
        super().__init__()
        self.T = T
        self.device = device
        self.initialize_alpha_beta_schedules()

        self.unet = UNet(
            in_ch=num_channels,
            base_channels=128,
            num_classes=num_classes,
            cond_dim=256
        ).to(device)

    def forward(self, x, t, y):
        return self.unet(x, t, y)  # predicted noise

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
    def p_sample(self, x_t, t, cond):
        one_minus_alpha_t = (1 - self.alphas[t])[:, None, None, None]
        sqrt_recip_alpha_t = (1.0 / torch.sqrt(self.alphas[t]))[:, None, None, None]
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        eps_theta = self.forward(x_t, t, cond)

        x = sqrt_recip_alpha_t * (
            x_t - (one_minus_alpha_t / sqrt_one_minus_alpha_hat_t) * eps_theta
        )

        if t[0].item() == 0:
            return x
        else:
            noise = torch.randn_like(x_t)
            return x + torch.sqrt(self.betas[t[0]]) * noise


def sample_image(model, img_shape, labels, device):
    n = len(labels)
    model.eval()
    with torch.no_grad():
        x_t = torch.randn((n, *img_shape), device=device)
        for t in reversed(range(model.T)):
            t_batch = torch.tensor([t] * n, device=device)
            x_t = model.p_sample(x_t, t_batch, labels)

    return x_t


if __name__ == "__main__":
    # Hyperparameters
    T = 1000
    B = 64
    lr = 1e-3
    epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    training_data = datasets.MNIST(root='./data_src', train=True, download=True,transform=transform)
    test_data = datasets.MNIST(root='./data_src', train=False, download=True, transform=transform)

    # training_data = datasets.CIFAR10(root='./data_src', train=True, download=True, transform=transform)
    # test_data = datasets.CIFAR10(root='./data_src', train=False, download=True, transform=transform)
    IMG_SHAPE = (1, 32, 32)
    training_dataloader = DataLoader(training_data,
                                     batch_size=B,
                                     shuffle=True,
                                     drop_last=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=B,
                                 shuffle=False,
                                 drop_last=True)
    test_loader_iter = iter(test_dataloader)

    model = SimpleDiffusionModel(T, num_channels=IMG_SHAPE[0], num_classes=10, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_writer = SummaryWriter(log_dir='runs/mnist/train')
    test_writer = SummaryWriter(log_dir='runs/mnist/test')

    step = 0
    for epoch in range(epochs):
        for index, (images, labels) in enumerate(training_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            model.train()
            t = torch.randint(0, T, (B,), device=device)

            actual_noise = torch.randn_like(images, device=device)

            xt = model.q_sample(images, t, actual_noise)  # noisy image
            predicted_noise = model(xt, t, labels)
            loss = model.loss(predicted_noise, actual_noise, writer=train_writer, index=step)

            if index % 100 == 0:
                with torch.no_grad():
                    try:
                        tbatch = next(test_loader_iter)
                        test_images, test_labels = tbatch
                    except StopIteration:
                        test_loader_iter = iter(test_dataloader)
                        test_images, test_labels = next(test_loader_iter)
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)

                    t_test = torch.randint(0, T, (test_images.shape[0],), device=device)
                    actual_noise_test = torch.randn_like(test_images, device=device)
                    xt_test = model.q_sample(test_images, t_test, actual_noise_test)  # noisy image
                    predicted_noise_test = model(xt_test, t_test, test_labels)
                    test_loss = model.loss(predicted_noise_test, actual_noise_test, writer=test_writer, index=step)

                    n = 8 # 8 samples to generate
                    images = sample_image(model, IMG_SHAPE, test_labels[:n], device)
                    log_images(test_writer, images, test_labels[:n], nrow=2, step=step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        print(f'Epoch [{epoch+1}/{epochs}] Step [{index+1}/{len(training_dataloader)}] Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'simple_diffusion_model.pth')
