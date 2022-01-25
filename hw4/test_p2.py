import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset_p2 import OfficeDataset
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':

    csv_path, data_dir, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    checkpoint_path = 'resnet-best.pth'

    test_transform = transforms.Compose([
                filenameToPILImage,
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    test_dataset = OfficeDataset(csv_path=csv_path, data_dir=data_dir, 
                                    label2id_path='label2id.csv', transform=test_transform, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 65)
    load_checkpoint(checkpoint_path, model)
    model.to(device)

    temp_correct = 0
    cnt = 0
    model.eval()
    predicts = []
    with torch.no_grad():
        for images in test_dataloader:
            images = images.to(device)
            logits = model(images)

            predict = torch.argmax(logits, dim=-1)
            predicts.extend(predict.cpu().tolist())

    # write to csv file
    id2label = {}
    for label, _id in test_dataset.label2id.items():
        id2label[_id] = label 

    fout = open(output_path, 'w')
    fout.write('id,filename,label\n')
    for i in range(len(predicts)):
        fout.write('{},{},{}\n'.format(i, test_dataset.data_df.iloc[i]['filename'], id2label[predicts[i]]))
           

