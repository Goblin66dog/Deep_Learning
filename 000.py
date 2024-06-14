from Data_Readers.Data_Reader import Dataset
from Data_Processors.Percent_Linear_Enhancement import PercentLinearEnhancement
import cv2
import numpy as np

name = ["image1", "image2", "label"]
for i in range(3):
    path = r"D:\Github_Repo\Open Pit Mining\validate_ori_image//"+name[i] +r"\99.TIF"
    item = Dataset(path)
    if i == 0:
        image = item.array
        image = image[:3]
        image = np.transpose(image, [1,2,0])
    elif i == 1:
        image = item.array
        image = PercentLinearEnhancement(image=image, clip_num=2, image_shape="HW").gray_process()
    else:
        image = item.array
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    path = r"D:\Github_Repo\Open Pit Mining\validate_ori_image//"+name[i] +r"\000.png"
    # cv2.imshow("a", image)
    # cv2.waitKey(0)
    cv2.imwrite(path, image)
