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

    # generate train.csv file (npy)
    with open("./csvs/train_%s_%s.csv" % (version, input_domain), "w") as f:
        train_csv = csv.writer(f, delimiter=";")
        for folder in train_folder:
            npys = sorted(glob(folder + "/*.npy"), key=lambda x: int(os.path.basename(x).replace(".npy", "")))
            for i, p in enumerate(npys[::3]):
                train_csv.writerow([npys[3 * i], npys[3 * i + 1], npys[3 * i + 2]])

    # generate test.csv file (npy)
    with open("./csvs/test_%s_%s.csv" % (version, input_domain), "w") as f:
        test_csv = csv.writer(f, delimiter=";")
        for folder in test_folder:
            npys = sorted(glob(folder + "/*.npy"), key=lambda x: int(os.path.basename(x).replace(".npy", "")))
            for i, p in enumerate(npys[::3]):
                test_csv.writerow([npys[3 * i], npys[3 * i + 1], npys[3 * i + 2]])


def generate_infer_set(folder_list, input_domain, version, desc="null"):
    save_path = "./csvs/infer_%s_%s_%s.csv" % (version, input_domain, desc)

    if os.path.exists(save_path):
        print("[%s] already exists" % save_path)
        raise Exception

    with open(save_path, "w") as f:
        infer_csv = csv.writer(f, delimiter=";")
        for folder in folder_list:
            npys = sorted(glob(folder + "/*.npy"), key=lambda x: int(os.path.basename(x).replace(".npy", "")))
            for i, p in enumerate(npys[:-2]):
                infer_csv.writerow([npys[i], npys[i + 1], npys[i + 2]])


if __name__ == "__main__":
    # folder_list = sorted(glob("/data3/interferogram_gan/retrieved/**/amplitude", recursive=True))
    # generate_train_test_set(folder_list, input_domain="amp", version="3")

    folder_list = sorted(glob("/data3/interferogram_gan/retrieved_clean/**/amplitude", recursive=True))
    generate_infer_set(folder_list, "amp", version="5", desc="cleanset")
