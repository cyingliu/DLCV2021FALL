import torch

from transformers import BertTokenizer
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import sys
import glob
import math

from catr.models import caption
from catr.datasets import coco, utils
from catr.configuration import Config

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(100)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def plot_attn_map(image_path, output_path, config, model, tokenizer, start_token, end_token):

    image = Image.open(image_path)
    original_image_shape = (np.array(image)).shape
    original_image = np.array(image)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)

    caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)

    image, caption, cap_mask = image.to(device), caption.to(device), cap_mask.to(device)

    output, attn_weights, sen_len = evaluate(config, model, image, caption, cap_mask)
    attn_weights = attn_weights.squeeze(0).cpu().numpy() # (128, 361)

    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=False)
    result = result.capitalize()
    print(result.capitalize().replace('[pad]', '').replace('[cls]', ''))

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(math.ceil(sen_len/5), 5)
    cnt = 0
    for y in range(math.ceil(sen_len/5)):
        for x in range(5):
            ax[y][x].axis('off')
            if cnt >= sen_len:
                continue
            if cnt == 0: 
                word = '<start>'
            elif cnt == sen_len - 1: 
                word = '<end>'
            else: 
                word = result.split(' ')[cnt].replace('.', '')
            ax[y][x].set_title(word)
            if cnt == 0:
                ax[y][x].imshow(original_image)
                cnt += 1
                continue
            attn_weight = attn_weights[cnt-1].reshape(19, 19)
            attn_weight = np.array(transforms.Compose([transforms.ToPILImage(), transforms.Resize((original_image_shape[0], original_image_shape[1]))])(attn_weight))
            ax[y][x].imshow(original_image)
            ax[y][x].imshow(attn_weight, alpha=0.5, cmap='rainbow', interpolation='nearest')
            cnt += 1
    plt.tight_layout()
    plt.savefig(output_path)

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template



@torch.no_grad()
def evaluate(config, model, image, caption, cap_mask):
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions, attn_weight = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption, attn_weight, i+1

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption




if __name__ == '__main__':
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    image_paths = glob.glob(os.path.join(input_dir, '*'))

    config = Config()
    model,_ = caption.build_model(config)
    checkpoint_path = 'weight493084032.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

    for image_path in image_paths:
        fname = image_path.replace(input_dir, '').replace('/', '').replace('.jpg', '.png')
        output_path = os.path.join(output_dir, fname)
        plot_attn_map(image_path, output_path, config, model, tokenizer, start_token, end_token)


