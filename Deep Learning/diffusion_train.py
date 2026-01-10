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


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    # def forward(self, x, c, t, context_mask):
    def forward(self, x, t, c):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on
        t = t.float()

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        # context_mask = context_mask[:, None]
        # context_mask = context_mask.repeat(1,self.n_classes)
        # context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        # c = c * context_mask

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        # print(t.shape, t.dtype)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


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



class SimpleDiffusionModel(nn.Module):
    def __init__(self, T, num_channels=3, num_classes=10, device="cpu"):
        super().__init__()
        self.T = T
        self.device = device
        self.initialize_alpha_beta_schedules()

        self.unet = ContextUnet(
            in_channels=num_channels,
            n_feat=128,
            n_classes=num_classes
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

    # transform = transforms.Compose([transforms.Resize((32, 32)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=(0.5,), std=(0.5,))])

    transform = transforms.Compose([transforms.ToTensor()])
    training_data = datasets.MNIST(root='./data_src', train=True, download=True,transform=transform)
    test_data = datasets.MNIST(root='./data_src', train=False, download=True, transform=transform)

    # training_data = datasets.CIFAR10(root='./data_src', train=True, download=True, transform=transform)
    # test_data = datasets.CIFAR10(root='./data_src', train=False, download=True, transform=transform)
    IMG_SHAPE = (1, 28, 28)
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

    train_writer = SummaryWriter(log_dir='runs/mnist/train_om')
    test_writer = SummaryWriter(log_dir='runs/mnist/test_om')

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
