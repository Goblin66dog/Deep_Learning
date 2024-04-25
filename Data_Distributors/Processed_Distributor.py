import copy
import glob
import os
import random


class Distributor:
    def __init__(self,train_pack_path):
        self.train_path_list = glob.glob(os.path.join(train_pack_path, "label/*"))
        self.validate_path_list = []

    @staticmethod
    def extract_name(file_path):
        name = os.path.splitext(os.path.basename(file_path))[0]
        return name

    def Random_Distributor(self, distribute_rate=0.1, processed_str= "_Processed"):
        validate_num = int(len(self.train_path_list) * distribute_rate)
        while len(self.validate_path_list) < validate_num:
            validate_path = random.choice(self.train_path_list)
            name = self.extract_name(validate_path)
            selector = name.find(processed_str)
            selector = name[:selector]
            for each_path in self.train_path_list[:]:
                if selector == self.extract_name(each_path)[:self.extract_name(each_path).find(processed_str)]:
                    self.validate_path_list.append(each_path)
                    self.train_path_list.remove(each_path)

        return self.train_path_list, self.validate_path_list

    def N_Cross_Distributor(self,N=3, processed_str= "_Processed"):
        selector_list = []
        for each_path in self.train_path_list:
            name = self.extract_name(each_path)
            selector = name.find(processed_str)
            selector = name[:selector]
            selector_list.append(selector)

        selector_list = list(set(selector_list))
        validate_path_list_container = copy.deepcopy(selector_list)

        N_cross_path_list = []
        each_cross_ave_num = round(len(selector_list) / N)
        sum_cross_num = 0
        bias = 1
        for each_cross in range(N):
            if each_cross == N - 1:
                each_cross_num = len(selector_list) - sum_cross_num
            else:
                each_cross_num = (each_cross_ave_num + bias)
                sum_cross_num += each_cross_num
                bias *= -1
            validate_list = random.sample(validate_path_list_container, each_cross_num)
            # train_list = list(set(selector_list).difference(set(validate_list)))
            validate_path_list_container = list(set(validate_path_list_container).difference(set(validate_list)))
            validate_path_list = []
            for path in self.train_path_list:
                for name in validate_list:
                    if name == self.extract_name(path)[:self.extract_name(path).find(processed_str)]:
                        validate_path_list.append(path)
            train_path_list = list(set(self.train_path_list).difference(set(validate_path_list)))
            N_cross_path_list.append([train_path_list, validate_path_list])
        return N_cross_path_list

if __name__ == "__main__":
    a = Distributor(r"D:\Project\CUMT_PAPER_DATASETS\P").N_Cross_Distributor()
    b1 = []
    for A, B in a:
        for bs in B:
            b1.append(bs)
    print(len(set(b1)))
    # print(len(a), len(b))

