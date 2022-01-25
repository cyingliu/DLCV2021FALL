import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pytorch_pretrained_vit import ViT


def load_checkpoint(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':
    
    checkpoint_path = 'result/p1/B16ft_trans_bs8_lr1e-2_warm0.2/vit_best.pth'
    model = ViT('B_16_imagenet1k')
    model.fc = nn.Linear(768, 37)
    load_checkpoint(checkpoint_path, model)
    positional_embedding = model.state_dict()['positional_embedding.pos_embedding'].squeeze(0).numpy()[1:, :] # (576, 768)
    row_norm = np.linalg.norm(positional_embedding, axis=1, keepdims=True) # (5776 1)
    normalize_positional_embedding = positional_embedding / row_norm
    cosine_smilarity = normalize_positional_embedding @ normalize_positional_embedding.T # (576, 576)
   
    fig, ax = plt.subplots(24, 24)
    cnt = 0
    for y in range(24):
        for x in range(24):
            im = ax[y][x].imshow(cosine_smilarity[cnt].reshape(24, 24), interpolation='nearest')
            ax[y][x].set_xticks([])
            ax[y][x].set_yticks([])
            ax[y][x].spines['top'].set_visible(False)
            ax[y][x].spines['right'].set_visible(False)
            ax[y][x].spines['bottom'].set_visible(False)
            ax[y][x].spines['left'].set_visible(False)
            if x == 0:
                ax[y][x].set_ylabel(str(y+1), rotation=0, fontsize=5, va='center')
            if y == 23:
                ax[y][x].set_xlabel(str(x+1), fontsize=5)
            cnt += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    fig.colorbar(im, cax=cbar_ax)
    fig.add_subplot(111, frameon=False)
    # # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Input patch column")
    plt.ylabel("Input patch row")
    plt.title("Visualization results")

    plt.savefig('cosine_smilarity.png')



