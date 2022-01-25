# TODO: create shell script for running your DANN model
# ! /bin/sh
# Example
wget https://github.com/grcgs2212/DLCV_HW3_3_Models/releases/download/0.0.0/dann_mnistm.pth
wget https://github.com/grcgs2212/DLCV_HW3_3_Models/releases/download/0.0.0/dann_usps.pth
python3 test_p3.py $1 $2 $3
