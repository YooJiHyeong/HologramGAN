import csv
import random
import os
from glob import glob

folder_list = sorted(glob("/data3/interferogram_gan/retrieved/**/amplitude", recursive=True))

print(len(folder_list))

# random folder shuffle
random.shuffle(folder_list)

# 5% of folders are test set
test_folder_num = len(folder_list) // 20

# dividing
test_folder = folder_list[:test_folder_num]
train_folder = folder_list[test_folder_num:]

print(len(test_folder), len(train_folder))

# generate train.csv file
with open("./csvs/train_3_amp_png.csv", "w") as f:
    train_csv = csv.writer(f, delimiter=";")
    for folder in train_folder:
        pngs = sorted(glob(folder + "/*.png"), key=lambda x: int(os.path.basename(x).replace(".png", "")))
        for i, p in enumerate(pngs[::3]):
            train_csv.writerow([pngs[3 * i], pngs[3 * i + 1], pngs[3 * i + 2]])

# generate test.csv file
with open("./csvs/test_3_amp_png.csv", "w") as f:
    test_csv = csv.writer(f, delimiter=";")
    for folder in test_folder:
        pngs = sorted(glob(folder + "/*.png"), key=lambda x: int(os.path.basename(x).replace(".png", "")))
        for i, p in enumerate(pngs[::3]):
            test_csv.writerow([pngs[3 * i], pngs[3 * i + 1], pngs[3 * i + 2]])

with open("./csvs/train_3_amp_npy.csv", "w") as f:
    train_csv = csv.writer(f, delimiter=";")
    for folder in train_folder:
        npys = sorted(glob(folder + "/*.npy"), key=lambda x: int(os.path.basename(x).replace(".npy", "")))
        for i, p in enumerate(npys[::3]):
            train_csv.writerow([npys[3 * i], npys[3 * i + 1], npys[3 * i + 2]])

# generate test.csv file
with open("./csvs/test_3_amp_npy.csv", "w") as f:
    test_csv = csv.writer(f, delimiter=";")
    for folder in test_folder:
        npys = sorted(glob(folder + "/*.npy"), key=lambda x: int(os.path.basename(x).replace(".npy", "")))
        for i, p in enumerate(npys[::3]):
            test_csv.writerow([npys[3 * i], npys[3 * i + 1], npys[3 * i + 2]])
