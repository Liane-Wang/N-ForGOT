import os
import sys
from pathlib import Path



dataset='yelp_clear'
model = 'TGAT' 
method = 'NForGOT'

devices = 0
n_epoch = 2
rp_times = 1

bs = 200  



## 7.27 set v_test as the stopper

filename_add = "test"
cmd = "WANDB_MODE=online python train.py --batch_size {} --dataset {} --model {} --method {} --n_epoch {}  --fuzzy_boundary 0 --rp_times {} "\
    .format(bs, dataset, model, method, n_epoch,  rp_times)
cmd += " --filename_add {}".format(filename_add)
cmd += " --device {}".format(devices)
cmd += " --rp_times {}".format(rp_times)
cmd += ' --stop_method {}'.format('acc')  # defaut: loss
os.system(cmd)