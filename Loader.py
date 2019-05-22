from PIL import Image

import torch
from torch.utils import data
from torchvision.transform import ToTensor


class CSVSet(data.Dataset):
    def __init__(self, csv_path, transform=None, aug_rate=0, delim=";"):
        self.items = []

        with open(csv_path, "r") as f:
            print("Path:", csv_path)
            lines = f.readlines()

            for line in lines:
                line = line.strip().split(",")
                self.items += [line]

    def __getitem__(self, idx):
        def _get(img):
            img = Image.open(img)
            img = ToTensor()(img)
            return img

        x1, x2, target = [_get(i) for i in self.items[idx]]
        
        x = torch.cat([x1, x2], 0)
        return x, target

    def __len__(self):
        return len(self.items)


def IVUSCSVLoader(csv_path, batch_size, sampler=False,
                  transform=None, aug_rate=0, num_workers=1,
                  shuffle=False, drop_last=False):

    dataset = CSVSet(csv_path, transform=transform, aug_rate=aug_rate, delim=",")
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)


if __name__ == "__main__":
    train = IVUSCSVLoader("/data2/JH/IVUS/train_2.csv", 8, mask=True)
    # print(len(train.dataset.items))
    # print("?", len(train))
    for img, target, mi, bc, path in train:
        print(img.shape, target, mi, bc, path)
        import torchvision

        ivus = img[:,0,:,:].unsqueeze(dim=1)
        mask = img[:,1,:,:].unsqueeze(dim=1)
        print(ivus.shape, mask.shape)
        ivus = (ivus - ivus.min()) / (ivus.max() - ivus.min())
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        torchvision.utils.save_image(ivus, "sample.png")
        torchvision.utils.save_image(mask, "mask_sample.png")
        input("waiting")
