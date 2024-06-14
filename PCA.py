import cv2
import numpy as np


from Data_Readers.Data_Reader import Dataset

X_item = Dataset(r"D:\Project\CUMT_PAPER_DATASETS\image1\09_10.TIF")
X = X_item.array
# X
# print("X=", X)
# Xi
X_mean = np.mean(X, axis=0)
X_centered = np.array([X_mean, X_mean, X_mean, X_mean])
print(X[:,:,:4].shape)
print(np.mean(X[0,0,:4]))
# print("Xi=", Xi)
# # Cov
# Cov = np.cov(Xi)

# import numpy as np
#
# def input_matrix():
#     rows = eval(input("rows num:"))
#     cols = eval(input("cols num:"))
#     X = []
#     for row in range(rows):
#         row_list = input("input"+str(row+1)+" row:")
#         row_list = row_list.split(',')
#         while len(row_list) != cols:
#             row_list = input("re_input"+str(row+1)+" row:")
#             row_list = row_list.split(',')
#         row_list = [int(i) for i in row_list]
#         X.append(row_list)
#     print(X)
#     return np.array(X)
#
#
# if __name__ == "__main__":
#     a = input_matrix()
#     b = input_matrix()
#     print("#####print matrix multiply########")
#     print(np.multiply(a,b))
#     print("#####print matrix det########")
