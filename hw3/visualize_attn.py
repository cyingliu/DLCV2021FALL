import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# from PyTorch_Pretrained_ViT.pytorch_pretrained_vit.model import ViT
from pytorch_pretrained_vit import ViT

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def load_checkpoint(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':

    fname = 'hw3_data/p1_data/val/31_4838.jpg'
    checkpoint_path = 'result/p1/B16ft_trans_bs8_lr1e-2_warm0.2/vit_best.pth'

    transform = transforms.Compose(
        [transforms.Resize((384, 384)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    transform_original = transforms.Resize((384, 384))
    
    img = Image.open(fname)
    orginal_img = np.array(transform_original(img))
    
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)


    model = ViT('B_16_imagenet1k')
    model.fc = nn.Linear(768, 37)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    model.to(device)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    for i in range(12):
        model.transformer.blocks[i].attn.proj_q.register_forward_hook(get_activation(f'transformer.blocks[{i}].attn.proj_q'))
        model.transformer.blocks[i].attn.proj_k.register_forward_hook(get_activation(f'transformer.blocks[{i}].attn.proj_k'))

    attn_maps = []
    with torch.no_grad():
        output = model(img)
        for i in range(12):
            query = activation[f'transformer.blocks[{i}].attn.proj_q'].squeeze() # (577, 768)
            key = activation[f'transformer.blocks[{i}].attn.proj_k'].squeeze() # (577, 768)
            attn = torch.matmul(query, torch.transpose(key, 0, 1)).cpu().numpy() # (577, 577)
            # attn = query @ key.T # (577, 577)
            attn_map = attn[0][1:].reshape(24, 24)
            attn_maps.append(attn_map)
        
        attn_maps = np.array(attn_maps) # (12, 24, 24)
        avg_attn_map = np.mean(attn_maps, axis=0)
        
        fig, ax = plt.subplots(1, 2)
        for a in ax:
            a.axis('off')
        ax[0].imshow(orginal_img)
        ax[0].set_title('Image')
        ax[1].imshow(avg_attn_map, cmap='rainbow', interpolation='nearest')
        ax[1].set_title('Visualization Result')
        plt.savefig('attention_map.png')

