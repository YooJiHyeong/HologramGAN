import csv
import random
import os
from glob import glob

def generate_train_test_set(folder_list, input_domain, version):
    print(len(folder_list))

    # random folder shuffle
    random.shuffle(folder_list)

    # 5% of folders are test set
    test_folder_num = len(folder_list) // 20

    # dividing
    test_folder = folder_list[:test_folder_num]
    train_folder = folder_list[test_folder_num:]

    print(len(test_folder), len(train_folder))

    # generate train.csv file (png)
    with open("./csvs/train_%s_%s_png.csv" % (version, input_domain), "w") as f:
        train_csv = csv.writer(f, delimiter=";")
        for folder in train_folder:
            pngs = sorted(glob(folder + "/*.png"), key=lambda x: int(os.path.basename(x).replace(".png", "")))
            for i, p in enumerate(pngs[::3]):
                train_csv.writerow([pngs[3 * i], pngs[3 * i + 1], pngs[3 * i + 2]])

    # generate test.csv file (png)
    with open("./csvs/test_%s_%s_png.csv" % (version, input_domain), "w") as f:
        test_csv = csv.writer(f, delimiter=";")
        for folder in test_folder:
            pngs = sorted(glob(folder + "/*.png"), key=lambda x: int(os.path.basename(x).replace(".png", "")))
            for i, p in enumerate(pngs[::3]):
                test_csv.writerow([pngs[3 * i], pngs[3 * i + 1], pngs[3 * i + 2]])

    # generate train.csv file (npy)
    with open("./csvs/train_%s_%s_npy.csv" % (version, input_domain), "w") as f:
        train_csv = csv.writer(f, delimiter=";")
        for folder in train_folder:
            npys = sorted(glob(folder + "/*.npy"), key=lambda x: int(os.path.basename(x).replace(".npy", "")))
            for i, p in enumerate(npys[::3]):
                train_csv.writerow([npys[3 * i], npys[3 * i + 1], npys[3 * i + 2]])

    # generate test.csv file (npy)
    with open("./csvs/test_%s_%s_npy.csv" % (version, input_domain), "w") as f:
        test_csv = csv.writer(f, delimiter=";")
        for folder in test_folder:
            npys = sorted(glob(folder + "/*.npy"), key=lambda x: int(os.path.basename(x).replace(".npy", "")))
            for i, p in enumerate(npys[::3]):
                test_csv.writerow([npys[3 * i], npys[3 * i + 1], npys[3 * i + 2]])


def generate_infer_set(folder_list, input_domain, version, desc="null"):
    png_path = "./csvs/infer_%s_%s_%s_png.csv" % (version, input_domain, desc)
    npy_path = "./csvs/infer_%s_%s_%s_npy.csv" % (version, input_domain, desc)

    if any([os.path.exists(png_path), os.path.exists(npy_path)]):
        print("[%s] or [%s] already exists" % (png_path, npy_path))
        raise Exception

    with open(png_path, "w") as f:
        infer_csv = csv.writer(f, delimiter=";")
        for folder in folder_list:
            pngs = sorted(glob(folder + "/*.png"), key=lambda x: int(os.path.basename(x).replace(".png", "")))
            for i, p in enumerate(pngs[:-2]):
                infer_csv.writerow([pngs[i], pngs[i + 1], pngs[i + 2]])

    with open(npy_path, "w") as f:
        infer_csv = csv.writer(f, delimiter=";")
        for folder in folder_list:
            npys = sorted(glob(folder + "/*.npy"), key=lambda x: int(os.path.basename(x).replace(".npy", "")))
            for i, p in enumerate(npys[:-2]):
                infer_csv.writerow([npys[i], npys[i + 1], npys[i + 2]])


if __name__ == "__main__":
    # folder_list = sorted(glob("/data3/interferogram_gan/retrieved_clean/**/amplitude", recursive=True))
    # generate_train_test_set(folder_list, input_domain="amp", version="3")

    folder_list = sorted(glob("/data3/interferogram_gan/retrieved_clean/**/phase", recursive=True))
    generate_infer_set(folder_list, "phase", version="3", desc="cleanset")
