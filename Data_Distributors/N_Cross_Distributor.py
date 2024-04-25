import copy
import glob
import os
import random


def Distributor(train_pack_path, N=3):
    train_path_list    = glob.glob(os.path.join(train_pack_path   , "label/*"))
    validate_path_list_container = copy.deepcopy(train_path_list)
    N_cross_path_list   = []
    each_cross_ave_num = round(len(train_path_list) / N)
    sum_cross_num  = 0
    bias = 1
    for each_cross in range(N):
        if each_cross == N-1:
            each_cross_num = len(train_path_list)-sum_cross_num
        else:
            each_cross_num = (each_cross_ave_num + bias)
            sum_cross_num += each_cross_num
            bias *= -1
        validate_list = random.sample(validate_path_list_container, each_cross_num)
        train_list = list(set(train_path_list).difference(set(validate_list)))
        validate_path_list_container = list(set(validate_path_list_container).difference(set(validate_list)))
        N_cross_path_list.append([train_list, validate_list])
    return N_cross_path_list
if __name__ == "__main__":
    for i in Distributor(r"D:\Project\CUMT_PAPER_DATASETS"):
        print(len(i[0]), len(i[1]))
