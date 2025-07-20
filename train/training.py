import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# -------------------- Dataset --------------------
class HangulDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        img = self.transform(img)
        label = self.labels[idx]
        one_hot = torch.zeros(len(self.classes))
        one_hot[label] = 1.0
        return img, one_hot

# -------------------- Models --------------------
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, num_classes, embed_size=50):
        super(Generator, self).__init__()
        self.label_emb = nn.Linear(num_classes, embed_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + embed_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        emb = self.label_emb(labels)
        x = torch.cat([noise, emb], dim=1)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, num_classes, embed_size=50):
        super(Discriminator, self).__init__()
        self.embed_size = embed_size
        self.label_embedding = nn.Linear(num_classes, embed_size)
        self.net = nn.Sequential(
            nn.Conv2d(nc + embed_size, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        batch_size, _, h, w = img.shape
        label_emb = self.label_embedding(labels)
        label_map = label_emb.view(batch_size, self.embed_size, 1, 1)
        label_map = label_map.expand(batch_size, self.embed_size, h, w)
        x = torch.cat([img, label_map], dim=1)
        return self.net(x).view(-1)

# -------------------- Utils --------------------
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# -------------------- Training Function --------------------
def train(data_root, epochs, batch_size, lr, nz, beta1, img_size, sample_class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HangulDataset(os.path.join(data_root, 'datasets'), image_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    num_classes = len(dataset.classes)
    class_to_idx = dataset.class_to_idx
    idx_to_class = dataset.idx_to_class

    nc, ngf, ndf = 1, 64, 64
    netG = Generator(nz, ngf, nc, num_classes).to(device)
    netD = Discriminator(nc, ndf, num_classes).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label, fake_label = 0.9, 0.0

    sample_class_idx = class_to_idx.get(sample_class) if sample_class in class_to_idx else None
    if sample_class_idx is not None:
        sample_label = torch.zeros(1, num_classes).to(device)
        sample_label[0, sample_class_idx] = 1.0

    for epoch in range(epochs):
        epoch_lossD, epoch_lossG = 0, 0
        for i, (imgs, labels) in enumerate(dataloader):
            b_size = imgs.size(0)
            imgs, labels = imgs.to(device), labels.to(device)

            # Discriminator
            netD.zero_grad()
            labels_real = torch.full((b_size,), real_label, device=device)
            output_real = netD(imgs, labels)
            errD_real = criterion(output_real, labels_real)
            errD_real.backward()

            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, labels)
            labels_fake = torch.full((b_size,), fake_label, device=device)
            output_fake = netD(fake.detach(), labels)
            errD_fake = criterion(output_fake, labels_fake)
            errD_fake.backward()
            optimizerD.step()

            # Generator
            netG.zero_grad()
            labels_trick = torch.full((b_size,), real_label, device=device)
            output_gen = netD(fake, labels)
            errG = criterion(output_gen, labels_trick)
            errG.backward()
            optimizerG.step()

            epoch_lossD += (errD_real + errD_fake).item()
            epoch_lossG += errG.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss_D: {epoch_lossD / len(dataloader):.4f} Loss_G: {epoch_lossG / len(dataloader):.4f}")

        os.makedirs(os.path.join(data_root, 'model_save'), exist_ok=True)
        torch.save(netG.state_dict(), os.path.join(data_root, 'model_save', f'G_epoch_{epoch+1}.pth'))
        torch.save(netD.state_dict(), os.path.join(data_root, 'model_save', f'D_epoch_{epoch+1}.pth'))

        if sample_class_idx is not None:
            with torch.no_grad():
                noise = torch.randn(1, nz, device=device)
                fake_sample = netG(noise, sample_label).detach().cpu()
                save_dir = os.path.join(data_root, 'samples', sample_class)
                os.makedirs(save_dir, exist_ok=True)
                utils.save_image(fake_sample, os.path.join(save_dir, f'sample_epoch_{epoch+1}.png'), normalize=True)

# -------------------- 실행 파라미터 설정 및 학습 실행 --------------------
if __name__ == '__main__':
    data_root = './'    # 데이터셋 루트 디렉토리
    epochs = 100         # 에포크 수
    batch_size = 128     # 배치 사이즈
    lr = 0.0002         # 학습률
    nz = 100            # 노이즈 벡터 크기
    beta1 = 0.5         # Adam 옵티마이저 beta1
    img_size = 64       # 이미지 크기
    sample_class = 'b0a1'  # 예시: '가' 클래스의 hex 이름. 샘플 저장 안할 시 None

    train(data_root, epochs, batch_size, lr, nz, beta1, img_size, sample_class)
