import os
import argparse

import torch
import torch.nn as nn
import torch.cuda as cuda
from torchvision.transforms import ToTensor, CenterCrop

from Runner import Runner
from Loader import CSVLoader
from Generator import Generator
from GeneratorWavpool import GeneratorWavpool
from GeneratorAvgpool import GeneratorAvgpool
from Discriminator import Discriminator
import utils


def arg_parse():
    desc = "HologramGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default=",".join(map(str, range(cuda.device_count()))),
                        help="Select GPU Numbering (Default : Maximum number of available GPUs)")
    parser.add_argument('--cpus', type=int, default="60",
                        help="The number of CPU workers")

    parser.add_argument('--csv_ver', type=str, default="3",
                        help='Select version of csv files for dataloader')
    parser.add_argument('--input_domain', type=str, required=True, choices=["amp", "phase", "ifgram"],
                        help='Select input domain')
    parser.add_argument('--file_ext', type=str, default="npy", choices=["png", "npy"],
                        help='Select input file extension')

    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory which models will be saved in')
    parser.add_argument('--load_last', action="store_true",
                        help='Load last model in save directory')
    parser.add_argument('--load_path', type=str,
                        help='Path of saved model')
    parser.add_argument('--load_abspath', type=str,
                        help='Absolute path of saved model')

    parser.add_argument('--inference', action="store_true",
                        help='Infer images')

    parser.add_argument('--total_step', type=int, default="10000000",
                        help="The number of total steps")

    parser.add_argument('--resl', type=int, default=384,    # default is 384 for phase / amplitude images
                        help="Resolution(i.e. shape) of images")

    parser.add_argument('--batch_train', type=int, default=88,
                        help="The number of batches for train")
    parser.add_argument('--batch_test', type=int, default=8,
                        help="The number of batches for test(note that this is equivalent to the number of logged sample images)")
    parser.add_argument('--batch_infer', type=int, default=128,
                        help="The number of batches for inference")

    parser.add_argument('--G', type=str, default="wavelet", choices=["unet", "wavelet", "avgpool"],
                        help="Select Generator")

    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parse()

    train_csv = "./csvs/train_%s_%s_%s.csv" % (arg.csv_ver, arg.input_domain, arg.file_ext)
    test_csv  = "./csvs/test_%s_%s_%s.csv"  % (arg.csv_ver, arg.input_domain, arg.file_ext)
    infer_csv = "./csvs/infer_%s_%s_%s.csv" % (arg.csv_ver, arg.input_domain, arg.file_ext)

    train_transform  = [CenterCrop(arg.resl), ToTensor()]
    test_transform   = [CenterCrop(arg.resl), ToTensor()]
    infer_transform  = [CenterCrop(arg.resl), ToTensor()]

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

    if arg.G == "unet":
        G = nn.DataParallel(Generator().to(device["model"]), output_device=device["output"])
    elif arg.G == "wavelet":
        G = nn.DataParallel(GeneratorWavpool().to(device["model"]), output_device=device["output"])
    elif arg.G == "avgpool":
        G = nn.DataParallel(GeneratorAvgpool().to(device["model"]), output_device=device["output"])

    D = nn.DataParallel(Discriminator().to(device["model"]), output_device=device["output"])

    train_loader = CSVLoader(train_csv, arg.batch_train, arg.file_ext, transform=train_transform, num_workers=arg.cpus, shuffle=True, drop_last=True)
    test_loader  = CSVLoader(test_csv,  arg.batch_test,  arg.file_ext, transform=test_transform,  num_workers=arg.cpus, shuffle=True, drop_last=True)

    tensorboard = utils.TensorboardLogger("%s/tb" % (arg.save_dir))

    runner = Runner(arg, total_step, G, D, train_loader, test_loader, device, tensorboard)

    if arg.inference:
        inference_loader = CSVLoader(infer_csv, arg.batch_infer, arg.file_ext, transform=infer_transform, num_workers=arg.cpus, shuffle=False, drop_last=False, cycle=False)

        runner.load(filename=arg.load_path, abs_filename=arg.load_abspath)
        runner.inference(inference_loader, arg.file_ext)
        exit()

    if any([arg.load_last, arg.load_path, arg.load_abspath]):
        runner.load(filename=arg.load_path, abs_filename=arg.load_abspath)

    runner.train()
