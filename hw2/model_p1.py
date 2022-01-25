import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        # (batch, 3, 64, 64)
        def conv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True))
        self.ls = nn.Sequential(
                nn.Conv2d(3, dim, 4, 2, 1), # (batch, 64, 32, 32)
                nn.LeakyReLU(0.2, inplace=True),
                conv_bn_relu(dim, 2 * dim), # (batch, 128, 16, 16)
                conv_bn_relu(2 * dim, 4 * dim), # (batch, 256, 8, 8)
                conv_bn_relu(4 * dim, 8 * dim), # (batch, 512, 4, 4)
                nn.Conv2d(dim * 8, 1, 4), # (batch, 1, 1, 1)
                nn.Sigmoid(),
            )
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

class Discriminator_wgan(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator_wgan, self).__init__()
        # (batch, 3, 64, 64)
        def conv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True))
        self.ls = nn.Sequential(
                nn.Conv2d(3, dim, 4, 2, 1), # (batch, 64, 32, 32)
                nn.LeakyReLU(0.2, inplace=True),
                conv_bn_relu(dim, 2 * dim), # (batch, 128, 16, 16)
                conv_bn_relu(2 * dim, 4 * dim), # (batch, 256, 8, 8)
                conv_bn_relu(4 * dim, 8 * dim), # (batch, 512, 4, 4)
                nn.Conv2d(dim * 8, 1, 4), # (batch, 1, 1, 1)
            )
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

class Generator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        # (batch, 100)
        # (batch, 100, 1, 1)
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True))
        # self.l1 = nn.Sequential(
        #     nn.Linear(in_dim, dim * 8 * 4 * 4), # (batch, 512 * 4 * 4)
        #     nn.BatchNorm1d(dim * 8 * 4 * 4),
        #     nn.LeakyReLU(0.2))
        self.l1 = nn.Sequential(
                nn.ConvTranspose2d(in_dim, dim * 8, 4, 1, 0),
                nn.BatchNorm2d(dim * 8),
                nn.ReLU(True),
            )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4), # (batch, 256, 8, 8)
            dconv_bn_relu(dim * 4, dim * 2), # (batch, 128, 16, 16)
            dconv_bn_relu(dim * 2, dim), # (batch, 64, 32, 32)
            nn.ConvTranspose2d(dim, 3, 4, 2, 1), # (batch, 3, 64, 64)
            nn.Tanh())
    def forward(self, x):
        y = self.l1(x)
        # y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

if __name__ == '__main__':
    
    from dataset_p1 import FaceDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision
    import torch
    
    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    data_dir = 'hw2_data/face/train/'
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    train_dataset = FaceDataset(data_dir, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
    generator = iter(train_dataloader)
    data = generator.next() # (batch, 3, 64, 64)
    # ================= model D ==================== #
    model_D = Discriminator()
    model_D.to(device)
    data = data.to(device)
    out = model_D(data) # (batch,)
    print(out.shape)
    model_D.cpu()

    # ================= model G ====================== #
    model_G = Generator(in_dim=100)
    model_G.to(device)
    z = torch.randn(4, 100).to(device)
    f_imgs = model_G(z)
    print(f_imgs.shape) # (batch, 3, 64, 64)
    f_sample = (f_imgs.data + 1) / 2.0
    filename = 'fake_sample.jpg'
    torchvision.utils.save_image(f_sample, filename, nrow=2)



