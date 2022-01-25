import torch.nn.functional as F
import torch.nn as nn
import torch

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cosine_similarity_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = F.cosine_similarity(a, b, dim=2)
    return logits

# version 1
class parametric_metric_1(nn.Module):
    def __init__(self, n_shot):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1600*(n_shot+1), 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, n_shot)
            )
    def forward(self, output, prototype):
        bs = output.shape[0]
        prototype = prototype.view(1, -1) # (1, 4800)
        prototype = prototype.repeat(bs, 1) # (n_query, 4800)
        merge_input = torch.cat((output, prototype), 1) # (bs, 6400)
        logits = self.fc(merge_input) # (bs, n_shot)
        return logits

# version 2
class parametric_metric(nn.Module):
    def __init__(self, n_shot):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1600*2, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
            )
    def forward(self, output, prototype):
        n_shot = prototype.shape[0]
        bs = output.shape[0]
        logits = []
        for n in range(n_shot):
            temp_prototype = prototype[n, :].repeat(bs, 1) # (bs, 1600)
            merge_input = torch.cat((output, temp_prototype), 1) # (bs, 3200)
            logits.append(self.fc(merge_input)) # (27, 1)
        logits = torch.cat(logits, 1) # (bs, n_shot)
        return logits

if __name__ == '__main__':
    parametric_metric_model = parametric_metric()
    a = torch.rand(27, 1600)
    b = torch.rand(5, 1600)
    logits = parametric_metric_model(a, b)
    print(logits.shape)
