import csv
import random
from glob import glob

folder_list = sorted(glob("../hologram/**/data3d", recursive=True))

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
train_csv = csv.writer(open("./train_2.csv", "w"), delimiter=",")
for f in train_folder:
    pngs = sorted(glob(f + "/*.png"))
    for i, p in enumerate(pngs[1::3]):
        train_csv.writerow([pngs[3 * i + 1], pngs[3 * i + 2], pngs[3 * i + 3]])

# generate test.csv file
test_csv = csv.writer(open("./test_2.csv", "w"), delimiter=",")
for f in test_folder:
    pngs = sorted(glob(f + "/*.png"))
    for i, p in enumerate(pngs[1::3]):
        test_csv.writerow([pngs[3 * i + 1], pngs[3 * i + 2], pngs[3 * i + 3]])
