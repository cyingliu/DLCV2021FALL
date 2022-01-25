#! /bin/sh
wget https://github.com/grcgs2212/DLCV_HW1-1_models/releases/download/0.0.0/dense161-best.pth
wget https://github.com/grcgs2212/DLCV_HW1-1_models/releases/download/0.0.0/dense169-best.pth
wget https://github.com/grcgs2212/DLCV_HW1-1_models/releases/download/0.0.0/resnet152-best.pth
wget https://github.com/grcgs2212/DLCV_HW1-1_models/releases/download/0.0.0/resnext-best.pth
wget https://github.com/grcgs2212/DLCV_HW1-1_models/releases/download/0.0.0/vgg16-best.pth
wget https://github.com/grcgs2212/DLCV_HW1-1_models/releases/download/0.0.0/vgg19_bn-best.pth
wget https://github.com/grcgs2212/DLCV_HW1-1_models/releases/download/0.0.0/vgg19_bn2-best.pth
wget https://github.com/grcgs2212/DLCV_HW1-1_models/releases/download/0.0.0/vgg19_bn3-best.pth
python3 test_p1.py $1 $2
