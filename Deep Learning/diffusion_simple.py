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


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
        )
        self.cond_fc = nn.Linear(cond_dim, out_ch)

    def forward(self, x, cond):
        skip = x
        h = self.model(x)
        cond = self.cond_fc(cond)[:, :, None, None]
        return h + cond, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_ch * 2, out_ch, 3, padding=1)
        self.cond_fc = nn.Linear(cond_dim, out_ch)

    def forward(self, x, cond, skip):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        h = self.conv(h)
        h = F.silu(h)
        cond = self.cond_fc(cond)[:, :, None, None]
        return h + cond


class SimpleNoisePredictor(nn.Module):
    def __init__(self, in_ch, base_ch, cond_dim, num_classes, diffusion_T=1000):
        super().__init__()
        self.cond_dim = cond_dim
        self.time_emb = nn.Embedding(diffusion_T, cond_dim)
        self.class_emb = nn.Embedding(num_classes, cond_dim)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1).to(device)
        self.encoder1 = EncoderBlock(base_ch, 2 * base_ch, cond_dim).to(device)
        self.encoder2 = EncoderBlock(2 * base_ch, base_ch * 4, cond_dim).to(device)
        self.encoder3 = EncoderBlock(base_ch * 4, base_ch * 8, cond_dim).to(device)

        self.decoder3 = DecoderBlock(base_ch * 8, base_ch * 4, cond_dim).to(device)
        self.decoder2 = DecoderBlock(base_ch * 4, base_ch * 2, cond_dim).to(device)
        self.decoder1 = DecoderBlock(base_ch * 2, base_ch, cond_dim).to(device)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1).to(device)


    def forward(self, x, t, y):
        '''
        x: (B, C, H, W) noisy image
        t: (B, ) tensor of timesteps
        y: (B, ) tensor of class labels
        '''
        cond = self.time_emb(t) + self.class_emb(y)

        h = self.in_conv(x)
        h, s1 = self.encoder1(h, cond)
        h, s2 = self.encoder2(h, cond)
        h, s3 = self.encoder3(h, cond)

        h = self.decoder3(h, cond, s3)
        h = self.decoder2(h, cond, s2)
        h = self.decoder1(h, cond, s1)
        h = self.out_conv(h)
        return h  # predicted noise


class SimpleDiffusionModel(nn.Module):
    def __init__(self, T, num_channels=3, num_classes=10, device="cpu"):
        super().__init__()
        self.T = T
        self.device = device
        self.initialize_alpha_beta_schedules()

        self.unet = SimpleNoisePredictor(
            in_ch=num_channels,
            base_ch=256,
            cond_dim=256,
            diffusion_T = T,
            num_classes=num_classes,
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
    B = 512
    lr = 1e-4
    epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])

    training_data = datasets.MNIST(root='./data_src', train=True, download=True,transform=transform)
    test_data = datasets.MNIST(root='./data_src', train=False, download=True, transform=transform)

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

    train_writer = SummaryWriter(log_dir='runs/mnist/train_simple_emb')
    test_writer = SummaryWriter(log_dir='runs/mnist/test_simple_emb')

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
