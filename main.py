import os
import argparse

import torch
import torch.nn as nn

from Runner import Runner
from Loader import CSVLoader
from Generator import Generator
from Discriminator import Discriminator
import utils


def arg_parse():
    desc = "PGGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                        help="Select GPU Numbering | 0,1,2,3,4,5,6,7 | ")
    parser.add_argument('--cpus', type=int, default="32",
                        help="The number of CPU workers")

    parser.add_argument('--save_dir', type=str, default='', required=True,
                        help='Directory which models will be saved in')

    parser.add_argument('--total_step', type=int, default="10000000",
                        help="The number of total steps")

    parser.add_argument('--batch_train', type=int, default="32",
                        help="The number of batches")

    parser.add_argument('--batch_test', type=int, default="8",
                        help="The number of batches")

    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parse()

    arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus

    device = {
        "model":  torch.device(0),
        "output": torch.device(1),
        "target": torch.device(1)
    }

    total_step = arg.total_step

    G = nn.DataParallel(Generator().to(device["model"]), output_device=device["output"])
    D = nn.DataParallel(Discriminator().to(device["model"]), output_device=device["output"])

    train_loader = CSVLoader("./train.csv", arg.batch_train, num_workers=arg.cpus, shuffle=True, drop_last=True)
    test_loader  = CSVLoader("./test.csv",  arg.batch_test,  num_workers=arg.cpus, shuffle=True, drop_last=True)

    tensorboard = utils.TensorboardLogger("%s/tb" % (arg.save_dir))

    runner = Runner(arg, total_step, G, D, train_loader, test_loader, device, tensorboard)
    runner.train()
