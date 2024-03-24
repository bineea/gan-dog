import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

# 路径设置（根据自己情况修改）
data_path = r'D:/Project/Python/gan-dog/data/img/dogs/'  # 数据集路径
model_path = './models/gan_dogs_model.pth'  # 模型保存路径
generated_images_path = './output'  # 生成图片保存路径

# 创建generated_images目录
os.makedirs(generated_images_path, exist_ok=True)

# 使用ImageFolder来读取数据
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
dataset = ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# 初始化生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5,0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5,0.999))

# 训练生成对抗网络
epochs = 500
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # 训练判别器
        discriminator.zero_grad()

        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_labels = torch.full((batch_size,), 1, dtype=torch.float, device=device)

        output = discriminator(real_images).squeeze()
        loss_real = criterion(output, real_labels)
        loss_real.backward()

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(noise)
        fake_labels = torch.full((batch_size,), 0, dtype=torch.float, device=device)

        output = discriminator(fake_images.detach()).squeeze()
        loss_fake = criterion(output, fake_labels)
        loss_fake.backward()

        optimizer_D.step()

        # 训练生成器
        generator.zero_grad()
        output = discriminator(fake_images).squeeze()
        loss_G = criterion(output, real_labels)
        loss_G.backward()
        optimizer_G.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}], D Loss: {loss_real.item() + loss_fake.item():.4f}, G Loss: {loss_G.item():.4f}")

# 保存模型
torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
}, model_path)


# 生成图片并保存
with torch.no_grad():
    noise = torch.randn(64, 100, 1, 1, device=device)
    fake_images = generator(noise)
    utils.save_image(fake_images, f"{generated_images_path}/dogs_{epoch}.png",
                     normalize=True, nrow=8)

# 打印出生成的图片
img = Image.open(f"{generated_images_path}/dogs_{epoch}.png")
plt.imshow(img)
plt.show()