import glob
import os
import random
def Distributor(train_pack_path, distribute_rate=.1):
    train_path_list = glob.glob(os.path.join(train_pack_path, "label/*"))
    validate_num = int(len(train_path_list) * distribute_rate)
    validate_path_list = random.sample(train_path_list, validate_num)
    train_path_list = list(set(train_path_list).difference(set(validate_path_list)))
    return train_path_list, validate_path_list

if __name__ == "__main__":
    A, B = Distributor(r"D:\Project\CUMT_PAPER_DATASETS")


