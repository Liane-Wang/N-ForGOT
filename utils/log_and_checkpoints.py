import logging
from pathlib import Path
import time
import os
import numpy as np
import torch

def set_logger(method,model, dataset, rp,filename, to_file):
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  timenow = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
  day = time.strftime("%Y%m%d", time.localtime(time.time()))
  based_path = "/export/data/liane/tgcl/OTG/log/{}/{}/{}/{}/{}".format(method,dataset,model,day,filename+timenow)
  Path(based_path).mkdir(parents=True, exist_ok=True)
  
  # Path("log/{}/{}/{}".format(method,model,day)).mkdir(parents=True, exist_ok=True)
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  if to_file:
    fh = logging.FileHandler(based_path +'/{}.log'.format(rp))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
  logger.info("Method:{}, Model:{}, Dataset:{}, Repeat time:{}".format(method,model, dataset,rp))
  return logger, timenow,day,based_path

def checkavaliable(args):
    if args.method == 'NForGOT' and args.dataset == 'yelp_clear':
        return True
    return False

def get_checkpoint_path(based_path,epoch, task,uml):
  
  p = f"{based_path}/task{task}/{epoch}.pth"

  if not os.path.exists(os.path.dirname(p)):
      os.makedirs(os.path.dirname(p), exist_ok=True)
  return p

class EarlyStopMonitor(object):
    def __init__(self, max_round, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(
        self,
        curr_val,
    ):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:  
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round

def set_model(args, task):
    mp = 'log/yelp_clear/' + str(task) +'.pth'
    model = torch.load(mp).to(args.device)
    return model