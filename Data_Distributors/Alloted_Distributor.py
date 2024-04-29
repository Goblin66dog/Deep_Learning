import glob
import os
def Distributor(train_pack_path, validate_pack_path):
    train_path_list    = glob.glob(os.path.join(train_pack_path   , "label/*"))
    validate_path_list = glob.glob(os.path.join(validate_pack_path, "label/*"))
    return train_path_list, validate_path_list

if __name__ == "__main__":
    A, B = Distributor(r"D:\Project\CUMT_PAPER_DATASETS", r"D:\Project\CUMT_PAPER_DATASETS_FINAL")
    print(len(A))
    print(len(B))