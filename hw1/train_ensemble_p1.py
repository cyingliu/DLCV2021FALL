import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

test_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(size=(40,40)),                                    
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize to (-1, 1)
])


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(50):
            filenames = sorted(glob.glob(os.path.join(root, '{}_*.png'.format(i))))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

def ensemble_selector(loss_function, y_hats, y_true, init_size=1,
                      replacement=True, max_iter=100):
    # Reference: https://github.com/DmitryBorisenko/ensemble_tutorial/blob/master/MNIST%20Ensembles.ipynb
    """Implementation of the algorithm of Caruana et al. (2004) 'Ensemble
    Selection from Libraries of Models'. Given a loss function mapping
    predicted and ground truth values to a scalar along with a dictionary of
    models with predicted and ground truth values, constructs an optimal
    ensemble minimizing ensemble loss, by default allowing models to appear
    several times in the ensemble.

    Parameters
    ----------
    loss_function: function
        accepting two arguments - numpy arrays of predictions and true values - 
        and returning a scalar
    y_hats: dict
        with keys being model names and values being numpy arrays of predicted
        values
    y_true: np.array
        numpy array of true values, same for each model
    init_size: int
        number of models in the initial ensemble, picked by the best loss.
        Default is 1
    replacement: bool
        whether the models should be returned back to the pool of models once
        added to the ensemble. Default is True
    max_iter: int
        number of iterations for selection with replacement to perform. Only
        relevant if 'replacement' is True, otherwise iterations continue until
        the dataset is exhausted i.e.
        min(len(y_hats.keys())-init_size, max_iter). Default is 100

    Returns
    -------
    ensemble_loss: pd.Series
        with loss of the ensemble over iterations
    model_weights: pd.DataFrame
        with model names across columns and ensemble selection iterations
        across rows. Each value is the weight of a model in the ensemble

    """
    # Step 1: compute losses
    losses = dict()
    for model, y_hat in y_hats.items():
        losses[model] = loss_function(y_hat, y_true)

    # Get the initial ensemble comprised of the best models
    losses = pd.Series(losses).sort_values()
    init_ensemble = losses.iloc[:init_size].index.tolist()

    # Compute its loss
    if init_size == 1:
        # Take the best loss
        init_loss = losses.loc[init_ensemble].values[0]
        y_hat_avg = y_hats[init_ensemble[0]].copy()
    else:
        # Average the predictions over several models
        y_hat_avg = np.array(
            [y_hats[mod] for mod in init_ensemble]).mean(axis=0)
        init_loss = loss_function(y_hat_avg, y_true)

    # Define the set of available models
    if replacement:
        available_models = list(y_hats.keys())
    else:
        available_models = losses.index.difference(init_ensemble).tolist()
        # Redefine maximum number of iterations
        max_iter = min(len(available_models), max_iter)

    # Sift through the available models keeping track of the ensemble loss
    # Redefine variables for the clarity of exposition
    current_loss = init_loss
    current_size = init_size

    loss_progress = [current_loss]
    ensemble_members = [init_ensemble]
    for i in range(max_iter):
        # Compute weights for predictions
        w_current = current_size / (current_size + 1)
        w_new = 1 / (current_size + 1)

        # Try all models one by one
        tmp_losses = dict()
        tmp_y_avg = dict()
        for mod in available_models:
            tmp_y_avg[mod] = w_current * y_hat_avg + w_new * y_hats[mod]
            tmp_losses[mod] = loss_function(tmp_y_avg[mod], y_true)

        # Locate the best trial
        best_model = pd.Series(tmp_losses).sort_values().index[0]

        # Update the loop variables and record progress
        current_loss = tmp_losses[best_model]
        loss_progress.append(current_loss)
        y_hat_avg = tmp_y_avg[best_model]
        current_size += 1
        ensemble_members.append(ensemble_members[-1] + [best_model])

        if not replacement:
            available_models.remove(best_model)

    # Organize the output
    ensemble_loss = pd.Series(loss_progress, name="loss")
    model_weights = pd.DataFrame(index=ensemble_loss.index,
                                 columns=y_hats.keys())
    for ix, row in model_weights.iterrows():
        weights = pd.Series(ensemble_members[ix]).value_counts()
        weights = weights / weights.sum()
        model_weights.loc[ix, weights.index] = weights

    return ensemble_loss, model_weights.fillna(0).astype(float)

def accuracy(predictions, targets, one_hot_targets=True):
    """Compute accuracy given arrays of predictions and targets.

    Parameters
    predictions: np.array
        (num examples, num_classes) of predicted class probabilities/scores
    targets: np.array
        (num examples, num_classes) of one hot encoded true class labels if
        'one_hot_targets' is True, or true class indices if it is False
    one_hot_targets: bool
        whether the target are in one-hot or class index format. Default is
        True

    Returns
    -------
    accuracy: float
        accuracy of predictions

    """
    if one_hot_targets:
        return (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()
    else:
        return (predictions.argmax(axis=1) == targets).mean()

if __name__ == '__main__':
    
    valid_path = 'p1_data/val_50'
    # load models for ensemble
    train_dataset = ImageDataset(valid_path, transform=test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=1)


    vgg16 = VGG16_custom()
    load_checkpoint('vgg16-best.pth', vgg16)

    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.fc = nn.Linear(2048, 50)
    load_checkpoint('resnet152-best.pth', resnet152)

    dense161 = torchvision.models.densenet161(pretrained=True)
    dense161.classifier = nn.Linear(2208, 50)
    load_checkpoint('dense161-best.pth', dense161)

    dense169 = torchvision.models.densenet169(pretrained=True)
    dense169.classifier = nn.Linear(1664, 50)
    load_checkpoint('dense169-best.pth', dense169)

    resnext = torchvision.models.resnext101_32x8d(pretrained=True)
    resnext.fc = nn.Linear(2048, 50)
    load_checkpoint('resnext-best.pth', resnext)

    vgg19_bn1 = torchvision.models.vgg19_bn(pretrained=True)
    vgg19_bn1.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 50)
    )
    load_checkpoint('vgg19_bn-best.pth', vgg19_bn1)

    vgg19_bn2 = torchvision.models.vgg19_bn(pretrained=True)
    vgg19_bn2.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 50)
    )
    load_checkpoint('vgg19_bn2-best.pth', vgg19_bn2)

    vgg19_bn3 = torchvision.models.vgg19_bn(pretrained=True)
    vgg19_bn3.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(2048, 50)
    )
    load_checkpoint('vgg19_bn3-best.pth', vgg19_bn3)

    models_dict = {'vgg16': vgg16, 'resnet152': resnet152, 'dense161': dense161, 'dense169': dense169, 'resnext': resnext,
            'vgg19_bn1': vgg19_bn1, 'vgg19_bn2': vgg19_bn2, 'vgg19_bn3': vgg19_bn3}
    y_hats = dict()
    flag = False
    for model_name, model in models_dict.items():
      print('Computing {}...'.format(model_name))
      model.to(device)
      model.eval()
      probs = []
      if not flag:
        labels = []
      with torch.no_grad():
          for data, target in train_dataloader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              # output = F.softmax(output, dim=-1)
              output = output.cpu().numpy()
              probs.append(output)
              if not flag:
                labels.append(target.view(-1, 1).cpu().numpy())

      probs = np.vstack(probs) # (22500, 50)
      if not flag:
        labels = np.vstack(labels).squeeze(-1) # (22500,)
        flag = True
      y_hats[model_name] = probs
      model.cpu()

    ensemble_acc, model_weights_acc = ensemble_selector(
        loss_function=lambda p, t: -accuracy(p, t, False),  # - for minimization
        y_hats=y_hats, y_true=labels,
        init_size=1, replacement=True, max_iter=50
    )
    model_weights_acc.to_csv('en_weight.csv')