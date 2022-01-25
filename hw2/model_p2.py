import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, dim=64, n_class=10):
        super(Discriminator, self).__init__()
        # (batch, 3, 64, 64)
        def conv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True))
        def conv_bn_relu_odd(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True))
        self.ls = nn.Sequential(
                nn.Conv2d(3, dim, 4, 2, 1), # (batch, 64, 14, 14)
                nn.LeakyReLU(0.2, inplace=True),
                conv_bn_relu(dim, 2 * dim), # (batch, 128, 7, 7)
                conv_bn_relu_odd(2 * dim, 4 * dim), # (batch, 256, 4, 4)
                conv_bn_relu(4 * dim, 8 * dim), # (batch, 512, 2, 2)
                nn.Conv2d(dim * 8, n_class+1, 2, 1, 0), # (batch, 11, 1, 1)
                nn.Sigmoid(),
            )

    def forward(self, x):
        y = self.ls(x)
        y = y.view(y.shape[0], -1) # (batch, 11)
        return y[:, 0], y[:, 1:] # real/fake, class 

class Generator(nn.Module):
    def __init__(self, in_dim=100, dim=64, emb_dim=50, latent_dim=384, n_class=10):
        super(Generator, self).__init__()
        # (batch, 100), (batch, 10)
        self.n_class = n_class
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True))

        def dconv_bn_relu_odd(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 3, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True))
        self.embedding = nn.Embedding(10, emb_dim)
        self.embedding_fc = nn.Linear(emb_dim, 1 * 1)
        self.latent_fc = nn.Linear(in_dim, latent_dim * 1 * 1)
        self.l1 = nn.Sequential(
                nn.ConvTranspose2d(latent_dim + 1, dim * 8, 2, 1, 0), # (batch, 512, 2, 2)
                nn.BatchNorm2d(dim * 8),
                nn.ReLU(True),
            )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4), # (batch, 256, 4, 4)
            dconv_bn_relu_odd(dim * 4, dim * 2), # (batch, 128, 7, 7)
            dconv_bn_relu(dim * 2, dim), # (batch, 64, 14, 14)
            nn.ConvTranspose2d(dim, 3, 4, 2, 1), # (batch, 3, 28, 28)
            nn.Tanh())
        self.latent_dim = latent_dim

    def forward(self, x, label):
        emb = self.embedding(label)
        fmap1 = self.embedding_fc(emb).view(x.shape[0], 1, 1, 1)
        fmap2 = self.latent_fc(x).view(x.shape[0], self.latent_dim, 1, 1)
        fmap = torch.cat([fmap1, fmap2], dim=1) # (batch, 385, 1, 1)
        # X = torch.cat((x, label_onehot), dim=1) # (batch, in_dim + 1)
        y = self.l1(fmap) # (batch, 512, 2, 2)
        # y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y
class Generator_fmnist(nn.Module):
    # REF: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/
    def __init__(self, emb_dim=50, latent_dim=384, n_channel=192):
        super(Generator_fmnist, self).__init__()
        # (batch, 100), (batch, 1)
        self.embedding = nn.Embedding(10, emb_dim)
        # self.embedding_fc = nn.Linear(emb_dim, 7 * 7)
        self.embedding_conv = nn.ConvTranspose2d(emb_dim, 1, 7, 1, 0) # (batch, 1, 7, 7)
        # self.latent_fc = nn.Linear(100, latent_dim * 7 * 7)
        self.latent_conv = nn.ConvTranspose2d(100, latent_dim, 7, 1, 0) # (batch, 384, 7, 7)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 1, n_channel, 5, 2, 2, 1), # (batch, 192, 14, 14)
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel, 3, 5, 2, 2, 1), # (batch, 3, 28, 28)

            nn.Tanh()
            )
        self.latent_dim = latent_dim
    def forward(self, x, label):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        label = label.view((label.shape[0]))
        emb = self.embedding(label)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1)
        fmap1 = self.embedding_conv(emb).view(x.shape[0], 1, 7, 7)
        fmap2 = self.latent_conv(x).view(x.shape[0], self.latent_dim, 7, 7)
        fmap = torch.cat([fmap1, fmap2], dim=1) # (batch, 385, 7, 7)
        out = self.main(fmap)
        return out

class Generator_big(nn.Module):

    def __init__(self):
        super(Generator_big, self).__init__()
        # (batch, 100), (batch, 10)
        def dconv(in_channel, out_channel):
            return nn.Sequential(
                    nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                )
        self.to_embedding = nn.Linear(10, 10)
        self.to_fmap = nn.Linear(110, 32 * 4 * 4)
        # self.to_embedding = nn.Linear(110, 32 * 4 * 4)
        self.dconv1 = dconv(32, 16)
        self.dconv2 = dconv(16, 8)
        self.dconv3 = dconv(8, 3)
        self.fc = nn.Sequential(
                nn.Linear(3 * 32 * 32, 3 * 28 * 28),
                nn.Tanh()
            )
    def forward(self, x, label_onehot):
        emb = self.to_embedding(label_onehot.float()) # (batch, 10)
        emb = torch.cat([x, emb], dim=1) # (batch, 110)
        fmap = self.to_fmap(emb) # (batch, 32 * 4 * 4)
        # emb = torch.cat([x, label_onehot.float()], dim=1)
        # fmap = self.to_embedding(emb)

        fmap = fmap.view(-1, 32, 4, 4)
        y = self.dconv1(fmap) # (batch, 16, 8, 8)
        y = self.dconv2(y) # (batch, 8, 16, 16)
        y = self.dconv3(y) # (batch, 3, 32, 32)
        y = y.view(-1, 3 * 32 * 32)
        out = self.fc(y)
        out = out.view(-1, 3, 28, 28)
        return out

class Discriminator_big(nn.Module):
    def __init__(self):
        super(Discriminator_big, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 3 * 32 * 32)
        def conv(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 4, 2, 1),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self.conv1 = conv(3, 8) # (batch, 8, 16, 16)
        self.conv2 = conv(8, 16) # (batch, 16, 8, 8)
        self.conv3 = conv(16, 32) # (batch, 32, 4, 4)
        self.fc2 = nn.Linear(32 * 4 * 4, 11)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: (batch, 3, 28, 28)
        x = self.fc1(x.view(-1, 3 * 28 * 28)).view(-1, 3, 32, 32) # (batch, 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 32 * 4 * 4)
        out = self.fc2(x)
        out = self.sigmoid(out)
        return out[:, 0], out[:, 1:]

class Discriminator_fmnist(nn.Module):
    # REF: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/
    def __init__(self, n_channel=16):
        super(Discriminator_fmnist, self).__init__()
        # (batch, 3, 28, 28)
        # (3, 2, 1) => floor((H+1)/2)
        # (3, 1, 1) => H
        self.main = nn.Sequential(
            nn.Conv2d(3, n_channel, 3, 2, 1), # (batch, 16, 14, 14)
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_channel, n_channel * 2, 3, 1, 1), # (batch, 32, 14, 14)
            nn.BatchNorm2d(n_channel * 2),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_channel * 2, n_channel * 4, 3, 2, 1), # (batch, 64, 7, 7)
            nn.BatchNorm2d(n_channel * 4),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_channel * 4, n_channel * 8, 3, 1, 1), # (batch, 128, 7, 7)
            nn.BatchNorm2d(n_channel * 8),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_channel * 8, n_channel * 16, 3, 2, 1), # (batch, 256, 4, 4)
            nn.BatchNorm2d(n_channel * 16),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_channel * 16, n_channel * 32, 3, 1, 1), # (batch, 512, 4, 4)
            nn.BatchNorm2d(n_channel * 32),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.fc = nn.Linear(n_channel * 32 * 4 * 4, 11)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.main(x)
        y = y.view(y.shape[0], -1)
        out = self.fc(y)
        out = self.sigmoid(out)
        return out[:, 0], out[:, 1:]

class Discriminator_small(nn.Module):
    # REF: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/
    def __init__(self, n_channel=16):
        super(Discriminator_small, self).__init__()
        # (batch, 3, 28, 28)
        # (3, 2, 1) => floor((H+1)/2)
        # (3, 1, 1) => H
        self.main = nn.Sequential(
            nn.Conv2d(3, n_channel, 3, 2, 1), # (batch, 16, 14, 14)
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_channel, n_channel * 2, 3, 2, 1), # (batch, 32, 7, 7)
            nn.BatchNorm2d(n_channel * 2),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_channel * 2, n_channel * 4, 3, 2, 1), # (batch, 64, 4, 4)
            nn.BatchNorm2d(n_channel * 4),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            )
        self.fc = nn.Linear(n_channel * 4 * 4 * 4, 11)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.main(x)
        y = y.view(y.shape[0], -1)
        out = self.fc(y)
        out = self.sigmoid(out)
        return out[:, 0], out[:, 1:]

class Discriminator_64(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator_64, self).__init__()
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
                nn.Conv2d(dim * 8, 11, 4), # (batch, 1, 1, 1)
                nn.Sigmoid(),
            )
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1, 11)
        return y[:, 0], y[:, 1:]

class Discriminator_32(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator_32, self).__init__()
        # (batch, 3, 28, 28)
        def conv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True))
        self.to_32 = nn.ConvTranspose2d(3, 3, 5, 1) # (batch, 3, 32, 32)
        self.ls = nn.Sequential(
                nn.Conv2d(3, dim, 4, 2, 1), # (batch, 64, 16, 16)
                nn.LeakyReLU(0.2, inplace=True),
                conv_bn_relu(dim, 2 * dim), # (batch, 128, 8, 8)
                conv_bn_relu(2 * dim, 4 * dim), # (batch, 256, 4, 4)
                nn.Conv2d(dim * 4, 11, 4), # (batch, 1, 1, 1)
                nn.Sigmoid(),
            )
    def forward(self, x):
        x = self.to_32(x)
        y = self.ls(x)
        y = y.view(-1, 11)
        return y[:, 0], y[:, 1:]

class Generator_64(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(Generator_64, self).__init__()
        # (batch, 100)
        # (batch, 100, 1, 1)
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True))
        self.to_embedding = nn.Linear(10, 20)
        self.l1 = nn.Sequential(
                nn.ConvTranspose2d(in_dim + 20, dim * 8, 4, 1, 0),
                nn.BatchNorm2d(dim * 8),
                nn.ReLU(True),
            )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4), # (batch, 256, 8, 8)
            dconv_bn_relu(dim * 4, dim * 2), # (batch, 128, 16, 16)
            dconv_bn_relu(dim * 2, dim), # (batch, 64, 32, 32)
            nn.ConvTranspose2d(dim, 3, 4, 2, 1), # (batch, 3, 64, 64)
            nn.Tanh())
    
    def forward(self, x, label_onehot):
        emb = self.to_embedding(label_onehot.float()).view(-1, 20, 1, 1) # (batch, 20, 1, 1)
        x = x.view(-1, 100, 1, 1)
        x = torch.cat([x, emb], dim=1) # (batch, 120, 1, 1)
        y = self.l1(x)
        # y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class Generator_32(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(Generator_32, self).__init__()
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
        self.to_embedding = nn.Linear(10, 20)
        self.l1 = nn.Sequential(
                nn.ConvTranspose2d(in_dim + 20, dim * 4, 4, 1, 0), # (batch, 256, 4, 4)
                nn.BatchNorm2d(dim * 4),
                nn.ReLU(True),
            )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 4, dim * 2), # (batch, 128, 8, 8)
            dconv_bn_relu(dim * 2, dim), # (batch, 64, 16, 16)
            nn.ConvTranspose2d(dim, 3, 4, 2, 1), # (batch, 3, 32, 32)
            nn.Conv2d(3, 3, 5, 1), # downsample to 28 (batch, 3, 28, 28)
            nn.Tanh())
    def forward(self, x, label_onehot):
        # (batch, 100), (batch, 10)
        emb = self.to_embedding(label_onehot.float()).view(-1, 20, 1, 1) # (batch, 20, 1, 1)
        x = x.view(-1, 100, 1, 1)
        x = torch.cat([x, emb], dim=1) # (batch, 120, 1, 1)
        y = self.l1(x)
        # y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

if __name__ == '__main__':
    
    from dataset_p2 import MNISTMDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision
    import torch
    
    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    data_dir = 'hw2_data/digits/mnistm/'
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )

    def label2onehot(label):
        # print(label.shape, label)
        label_onehot = torch.zeros(label.shape[0], 10).to(device)
        label_onehot = label_onehot.scatter_(1, label, 1).view(label.shape[0], 10)
        return label_onehot
    # downsample_transform = transforms.Compose(
    #     [transforms.ToPILImage(),
    #     transforms.Resize((28, 28)),
    #     transforms.ToTensor()])


    train_dataset = MNISTMDataset(data_dir, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
    generator = iter(train_dataloader)
    img, label = generator.next() # (batch, 3, 64, 64)
    img, label = img.to(device), label.to(device)

    # ================= check downsample =========== #
    # imgs_sample = (img + 1) / 2.0
    # filename = 'sample_up_digit.jpg'
    # torchvision.utils.save_image(imgs_sample, filename, nrow=2)
    # imgs_downsample = downsample_transform(imgs_sample)
    # filename = 'sample_down_digit.jpg'
    # torchvision.utils.save_image(imgs_downsample, filename, nrow=2)
    # ================= model D ==================== #
    model_D = Discriminator_32()
    model_D.to(device)
    out_dis, out_class = model_D(img) # (batch,)
    print("D output:", out_dis.shape, out_class.shape)
    model_D.cpu()

    # ================= model G ====================== #
    model_G = Generator_32()
    model_G.to(device)
    z = torch.randn(4, 100).to(device)
    rand_label = torch.randint(0, 10, (4, 1)).long().to(device)
    rand_label_onehot = label2onehot(rand_label)
    print('label onehot:', rand_label_onehot.shape)
    f_imgs = model_G(z, rand_label_onehot)
    print(f'G output:', f_imgs.shape) # (batch, 3, 64, 64)
    f_sample = (f_imgs.data + 1) / 2.0
    filename = 'fake_sample.jpg'
    torchvision.utils.save_image(f_sample, filename, nrow=2)




