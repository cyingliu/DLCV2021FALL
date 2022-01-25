import torch.nn as nn
import torch

class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()

        def conv_block(in_channels, out_channels):
            bn = nn.BatchNorm2d(out_channels)
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                bn,
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
            )
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


if __name__ == '__main__':

    from dataset import MiniDataset
    from sampler import PrototypicalSampler
    from torch.utils.data import DataLoader
    import random
    import numpy as np
    from utils import euclidean_metric

    train_dataset = MiniDataset(csv_path='hw4_data/mini/train.csv', data_dir='hw4_data/mini/train/')

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)


    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    N_way = 3
    N_shot = 1
    N_query = 3
    train_loader = DataLoader(
        train_dataset,
        num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn,
        batch_sampler=PrototypicalSampler(label=train_dataset.data_df['label'].tolist(), 
                                    iteration=1, class_per_it=N_way, sample_per_class=N_shot+N_query))
    model = Convnet()
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for i, (img, label) in enumerate(train_loader):
        # img: (30, 3, 84, 84), output: (30, 1600)
       img = img.to(device)
       data_support, data_query = img[:N_way*N_shot], img[N_way*N_shot:]
       print(img.shape)
       
       # create the relative label (0 ~ N_way-1) for query data
       label_encoder = {label[i * N_shot] : i for i in range(N_way)}
       query_label = torch.LongTensor([label_encoder[class_name] for class_name in label[N_way * N_shot:]]).to(device)

       prototype = model(data_support) 
       prototype = prototype.reshape(N_shot, N_way, -1).mean(dim=0) # (3, 1600)

       output = model(data_query)
       logits = euclidean_metric(output, prototype)
       loss = criterion(logits, query_label)

       predict = torch.argmax(logits, dim=-1)
       accuracy = torch.sum(predict == query_label.view(predict.shape)).item() / data_query.shape[0]
       print(loss.item())
       print(accuracy)



       # output = model(img)
