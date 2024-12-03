import numpy as np
import torch
import argparse
import shutil
import os

from src import config
from src.slam import SLAM
from src.datasets import get_dataset


import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    setup_seed(43)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='Path to config file.')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--max_frames", type=int, default=-1, help="Only [0, max_frames] Frames will be run")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")

    cfg = config.load_config(
        args.config, './configs/base.yaml'
    )

    assert cfg['mode'] in ['rgbd', 'mono', 'stereo'], cfg['mode']
    print(f"\n\n** Running {cfg['data']['input_folder']} in {cfg['mode']} mode!!! **\n\n")

    print(args)
    output_dir = cfg['data']['output']

    dataset = get_dataset(cfg, args, device=args.device)

    slam = SLAM(args, cfg)
    slam.run(dataset)
    

    print('Done!')

