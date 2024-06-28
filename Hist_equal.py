'''
Author: Yanxin_Jiang
Date: 2022-05-08 19:48:21
LastEditors: Yanxin_Jiang
LastEditTime: 2022-06-22 09:49:55
Description: 

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
from turtle import delay
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import codecs
import SimpleITK as sitk
import cv2
import pydicom
# m = 100
# x = np.linspace(-1, 1, m)
# x0 = np.linspace(-1, 1, 100)
# y_exact = 1 + 2 * x
# xi = x + np.random.normal(0, 0.05, 100)
# x1 = x0 + np.random.normal(0, 0.05, 100)
# yi = 1 + 2 * x1 + np.random.normal(0, 0.05, 100)
# A = np.vstack([xi**0, xi**1])
# print(A.T.shape, '\n', yi.shape )
# sol, r, rank, s = la.lstsq(A.T, yi)   #求取各个系数大小
# y_fit = sol[0] + sol[1] * x
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(xi, yi, 'go', alpha=0.5, label='Simulated data')
# ax.plot(x, y_exact, 'k', lw=2, label='True value y = 1 + 2x')
# ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
# ax.set_xlabel("x", fontsize=18)
# ax.set_ylabel('y', fontsize=18)
# ax.legend(loc=2)         #设置曲线标注位置
# plt.show()

# x = np.arange(33)
# x = np.array(x)
# y = np.zeros((12396, 33)).T 
# print('x.shape :', x.shape, '\ny.shape :', y.shape)
# z = np.dot(x, y)
# print('z:', z, '\nz.shape', z.shape)

def GetWorldPointFrom2DTrans():
    file = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\transformix_result/outputpoints.txt'
    f = codecs.open(file, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
    line = f.readline()   # 以行的形式进行读取文件
    a = []
    A = []
    list1 = []
    list2 = []
    while line:
        a = line.split()
        b = float(a[27])  # 这是选取需要读取的位数
        c = float(a[28])
        list1.append(b)  # 将其添加在列表之中
        list2.append(c)  # 将其添加在列表之中
        line = f.readline()
    f.close()
    X = np.array(list1)
    Y = np.array(list2)
    return X, Y


def limitedEqualize(img_array, limit = 0.01):
    clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8,8))
    return clahe.apply(img_array)

def global_linear_transmation(img): #将灰度范围设为0~255

    maxV=img.max()
    minV=img.min()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = ((img[i,j]-minV)*255)/(maxV-minV)

    return img

if __name__ =='__main__':
    # x, y = GetWorldPointFrom2DTrans()
    # print('x:', x, '\ny:', y, '\nx.shape:', x.shape, '\ny.shape', y.shape)
    i_pati = r'E:\AAAA\AAAA\Science\Scoliosis\Patient\Wuyifan\X/wuyifan.nii'
    sitk_img = sitk.ReadImage(i_pati)
    img_array = sitk.GetArrayFromImage(sitk_img)    
    img_array = np.squeeze(img_array)
    Spacing = sitk_img.GetSpacing()
    plt.subplot(121)
    plt.imshow(img_array,'gray')
    plt.title('origin')

    imgarray_0255 = global_linear_transmation(img_array)
    imgarray_0255 = np.array(imgarray_0255, dtype='uint8')
    #imgarray_0255 = img_array.copy()
    imgarray_hist = cv2.equalizeHist(imgarray_0255)
    hist_img = sitk.GetImageFromArray(imgarray_hist)
    hist_img.SetSpacing(Spacing)
    # sitk_hisequal = sitk.AdaptiveHistogramEqualizationImageFilter()
    # sitk_hisequal.SetAlpha(0.9)
    # sitk_hisequal.SetBeta(0.9)
    # sitk_hisequal.SetRadius(3)
    # sitk_hisequal = sitk_hisequal.Execute(sitk_img)
    # histimg_array = sitk.GetArrayFromImage(sitk_hisequal)    
    # histimg_array = np.squeeze(histimg_array)
    plt.subplot(122)
    plt.imshow(imgarray_hist,'gray')
    plt.title('histeq')

    plt.show()
    delay(100)
    save_path = r'E:\AAAA\AAAA\Science\Scoliosis\Patient\Wuyifan\X'
    file_name = save_path + '/WYF_hist.nii'
    sitk.WriteImage(hist_img,file_name)
    