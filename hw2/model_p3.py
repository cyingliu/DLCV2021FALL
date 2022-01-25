import torch.nn as nn
import torch
import torchvision.models as models

class GradReverse(torch.autograd.Function):
    # https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):

    def __init__(self):
        super(DANN, self).__init__()
        backbone = models.resnet50(pretrained=False)
        self.feature_extractor = nn.Sequential(*(list(backbone.children())[:-1]))
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.Sigmoid()
            )
    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 2048)
        reverse_feature = GradReverse.apply(feature)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output

class DANN_small(nn.Module):
    # (batch, 3, 64, 64)
    def __init__(self):
        super(DANN_small, self).__init__()
        def conv_block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True)
                )
        self.feature_extractor = nn.Sequential(
            conv_block(3, 64), # (batch, 64, 32, 32)
            conv_block(64, 128), # (batch, 128, 16, 16),
            conv_block(128, 256), # (batch, 256, 8, 8),
            conv_block(256, 512), # (batch, 512, 4, 4)
            conv_block(512, 1024), # (batch, 1024, 2, 2)
            nn.AdaptiveAvgPool2d(1) # (batch, 1024, 1, 1)
            )
        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.Sigmoid()
            )
    def forward(self, x, alpha):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 1024)
        reverse_feature = GradReverse.apply(feature)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output

class DANN_small_domain3_alpha(nn.Module):
    # (batch, 3, 64, 64)
    def __init__(self):
        super(DANN_small_domain3_alpha, self).__init__()
        def conv_block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True)
                )
        # def conv_notpool_block(in_channel, out_channel):
        #     return nn.Sequential(
        #         nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        #         nn.BatchNorm2d(out_channel),
        #         nn.ReLU(inplace=True)
        #         )
        self.feature_extractor = nn.Sequential(
            conv_block(3, 64), # (batch, 64, 32, 32)
            conv_block(64, 128), # (batch, 128, 16, 16),
            conv_block(128, 256), # (batch, 256, 8, 8),
            conv_block(256, 512), # (batch, 512, 4, 4)
            conv_block(512, 1024), # (batch, 1024, 2, 2)
            nn.AdaptiveAvgPool2d(1) # (batch, 1024, 1, 1)
            )
        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.Sigmoid()
            )
    def forward(self, x, alpha):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 1024)
        # reverse_feature = GradReverse.apply(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output

class DANN_tiny_bn_dp(nn.Module):
    # (batch, 3, 64, 64)
    # (batch, 3, 28, 28)
    def __init__(self):
        super(DANN_tiny_bn_dp, self).__init__()
        def conv_block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.Dropout2d(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
                )
        self.feature_extractor = nn.Sequential(
            conv_block(3, 32), # (batch, 32, 32, 32) # (batch, 32, 14, 14)
            conv_block(32, 48), # (batch, 64, 16, 16),
            conv_block(48, 48), # (batch, 48, 8, 8),
            # nn.AdaptiveAvgPool2d(1) # (batch, 1024, 1, 1)
            )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48*8*8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(48*8*8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.Sigmoid()
            )
    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 48*8*8)
        reverse_feature = GradReverse.apply(feature)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output

class DANN_micro(nn.Module):
    # (batch, 3, 28, 28)
    def __init__(self):
        super(DANN_micro, self).__init__()

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, 5), # (bs, 32, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(2), # (bs, 32, 14, 14)

                nn.Conv2d(32, 48, 5), # (bs, 50, 14, 14)
                nn.BatchNorm2d(48),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2) # (bs, 50, 7, 7)

            )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Sigmoid()
            )
    def forward(self, x, alpha):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 48*4*4)
        reverse_feature = GradReverse.apply(feature)
        # reverse_feature = ReverseLayerF.apply(feature, alpha)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output

class DANN_svhn1(nn.Module):
    # (batch, 3, 28, 28)
    def __init__(self):
        super(DANN_svhn1, self).__init__()

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), # (bs, 32, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(2), # (bs, 32, 14, 14)

                nn.Conv2d(32, 48, 3, 1, 1), # (bs, 50, 14, 14)
                nn.BatchNorm2d(48),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2) # (bs, 50, 7, 7)

            )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48*7*7, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(48*7*7, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Sigmoid()
            )
    def forward(self, x, alpha):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 48*7*7)
        reverse_feature = GradReverse.apply(feature)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return label_output, domain_output

class DANN_svhn2(nn.Module):
    # (batch, 3, 28, 28)
    def __init__(self):
        super(DANN_svhn2, self).__init__()

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), # (bs, 32, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(2), # (bs, 32, 14, 14)

                nn.Conv2d(32, 48, 3, 1, 1), # (bs, 50, 14, 14)
                nn.BatchNorm2d(48),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2) # (bs, 50, 7, 7)

            )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48*7*7, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(48*7*7, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Sigmoid()
            )
    def forward(self, x, alpha):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 48*7*7)
        reverse_feature = GradReverse.apply(feature)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return label_output, domain_output

class DANN_svhn3(nn.Module):
    # (batch, 3, 28, 28)
    def __init__(self):
        super(DANN_svhn3, self).__init__()

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), # (bs, 32, 28, 28)
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2), # (bs, 32, 14, 14)

                nn.Conv2d(32, 48, 3, 1, 1), # (bs, 50, 14, 14)
                nn.BatchNorm2d(48),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2) # (bs, 50, 7, 7)

            )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48*7*7, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(48*7*7, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Sigmoid()
            )
    def forward(self, x, alpha):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 48*7*7)
        reverse_feature = GradReverse.apply(feature)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return label_output, domain_output

class DANN_svhn4(nn.Module):
    # (batch, 3, 28, 28)
    def __init__(self):
        super(DANN_svhn4, self).__init__()

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, 5), # (bs, 32, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(2), # (bs, 32, 14, 14)

                nn.Conv2d(32, 48, 5), # (bs, 50, 14, 14)
                nn.BatchNorm2d(48),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2) # (bs, 50, 7, 7)

            )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
            )
        self.label_classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Sigmoid()
            )
    def forward(self, x, alpha):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, 48*4*4)
        reverse_feature = GradReverse.apply(feature)
        # reverse_feature = ReverseLayerF.apply(feature, alpha)

        label_output = self.label_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output
if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    from dataset_p3 import DigitDataset
    import torchvision.transforms as transforms

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    data_dir = 'hw2_data/digits/mnistm/'
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         # transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # for usps
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ] )

    train_dataset = DigitDataset(data_dir, transform, mode='train', domain='source')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)

    generator = iter(train_dataloader)
    img, class_label = generator.next()
    img, class_label = img.to(device), class_label.to(device)

    domain_label = torch.ones((4, 1)).to(device)

    model = DANN_micro()
    model.to(device)
    class_logit, domain_logit = model(img) # (4, 10), (4, 1)
    criterion_domain = nn.BCELoss()
    criterion_class = nn.CrossEntropyLoss()
    print(class_logit.shape)
    print(domain_logit.shape)
    loss_domain = criterion_domain(domain_logit, domain_label)
    loss_class = criterion_class(class_logit, class_label)
    predict = torch.argmax(class_logit, dim=-1)
    correct = torch.sum(predict == class_label.view(predict.shape)).item()
    print('correct', correct)



