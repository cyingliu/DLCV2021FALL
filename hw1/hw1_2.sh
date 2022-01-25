#!/bin/bash
wget https://github.com/grcgs2212/DLCV-HW1_2-models/releases/download/0.0.0/fcn8s-best.pth
python3 test_p2.py $1 $2
