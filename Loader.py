import random
from PIL import Image

import torch
from torch.utils import data
from torchvision.transforms import ToTensor, Resize


class CSVSet(data.Dataset):
    def __init__(self, csv_path, transform=None, aug_rate=0, delim=";"):
        self.items = []

        with open(csv_path, "r") as f:
            print("Path:", csv_path)
            lines = f.readlines()

            for line in lines:
                line = line.strip().split(",")
                self.items += [line]

        self.transform = [Resize((512, 512)), ToTensor()]
        if transform is not None:
            self.transform[1:1] = transform
        
    def __getitem__(self, idx):
        def _get(img):
            img = Image.open(img)
            for t in self.transform:
                img = t(img)
            return img

        x1, x2, target = [_get(i) for i in self.items[idx]]
        x = torch.cat([x1, x2], 0)

        path = self.items[idx][1]
        return x, target, path

    def __len__(self):
        return len(self.items)


def CSVLoader(csv_path, batch_size, sampler=False,
              transform=None, aug_rate=0, num_workers=1,
              shuffle=False, drop_last=False):

    def _cycle(loader):
        while True:
            for element in loader:
                yield element
            random.shuffle(loader.dataset.items)
        
    dataset = CSVSet(csv_path, transform=transform, aug_rate=aug_rate, delim=",")
    loader = _cycle(data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True))

    return loader


if __name__ == "__main__":
    # train_loader = CSVLoader("./train.csv", arg.batch_train, num_workers=arg.cpus, shuffle=True, drop_last=True)
    test_loader  = CSVLoader("./test.csv",  8,  num_workers=40, shuffle=True, drop_last=True)
    # print(len(test_loader.dataset.items))
    print("======================================")
    for i, (img, target, path) in enumerate(test_loader):
        print(i)
        # print(img.shape, target.shape)

        # import torchvision

        # i1 = img[0, 0]
        # i2 = img[0, 1]
        # t = target[0, 0]

        # # ivus = (ivus - ivus.min()) / (ivus.max() - ivus.min())
        # # mask = (mask - mask.min()) / (mask.max() - mask.min())
        # torchvision.utils.save_image(i1, "i1.png")
        # torchvision.utils.save_image(i2, "i2.png")
        # torchvision.utils.save_image(t, "t.png")
        # input("wating")