'''
Author: Yanxin_Jiang
Date: 2022-04-28 16:48:03
LastEditors: Yanxin_Jiang
LastEditTime: 2022-07-15 15:23:29
Description: 
            对数字人体模型进行X光投影, 并保存投影文件为nii格式
Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
from ctypes import *
import numpy as np
import time
import cv2 
import matplotlib.pyplot as plt
import SimpleITK as sitk


"""     
    obj文件共有三类组织：
        1.skin有****个vertices，****个meshes
        2.skeleton有****个vertices，****个meshes
        3.lungs有7113个****，****个meshes
    共有294070个vertices，582342个meshes
    
"""
#plt.switch_backend('agg')

# ---------------------------------1.读取数字人躯干模型obj文件的点数据及面数据-------------------------------
Obj_filepath = "E:/AAAA/AAAA/SeniorYear/digital_human_Xray/obj/"
Obj_file = Obj_filepath + 'Mean_Torso.obj'

def setWindowLevel(img, windowMid=0, windowwidth=100):
    img_filter = sitk.IntensityWindowingImageFilter()
    # img_filter.SetOutputMaximum()
    # img_filter.SetOutputMinimum()
    windowMax = windowMid + windowwidth/2.0
    windowMin = windowMid - windowwidth/2.0
    img_filter.SetWindowMinimum(windowMin)
    img_filter.SetWindowMaximum(windowMax)

    return img_filter.Execute(img)


def ConePrj_Mesh(Verts_point, TRI_point, Dtctr_in_world = [0. ,330. ,-100.],
                 xvec_in_world = [0 ,0 ,-1], yvec_in_world = [1 ,0 ,0], 
                 Src_in_world = [-4.667 ,450 ,-30.]):
    # # 逐行获取Verts和TRI
    # with open(file_path) as file:
    #     verts = []
    #     tri = []    
    #     while 1:
    #         line = file.readline()   
    #         if not line:
    #             break
    #         strs = line.split()
    #         if strs[0] == 'v':
    #             verts.append((float(strs[1]),float(strs[2]),float(strs[3])))            
    #         if strs[0] == 'f':
    #             tri.append((float(strs[1])-1.,float(strs[2])-1.,float(strs[3])-1.))            
  
    # Verts_point = np.array(verts)
    # TRI_point = np.array(tri)
   
    """
    #存入Verts_point和TRI_point
    np.savetxt('Verts_point.txt',Verts_point)
    np.savetxt('TRI_point.txt',TRI_point)
    """
    # 三类顶点各项总数
    n_Verts = Verts_point.shape[0]
    # print('n_Verts = ',n_Verts)
    n_TRI = TRI_point.shape[0]
    # print('n_TRI = ',n_TRI)
    """
    #读取Verts_point和TRI_point
    Verts_point = np.loadtxt('Verts_point.txt')
    TRI_point = np.loadtxt('TRI_point.txt')
    """
    # ---------------------------------2.创建输入输出参数的C++类型数组------------------------------- 
    # verts_array = [[0 for i in range(294070)]for j in range(3)]
    # tri_array = [[0 for i in range(582342)]for j in range(3)]

    #[列，行]矩形探测器在世界坐标系中的尺寸
    [n_col, n_row] = [500, 950] 
    # Src x射线源在世界坐标系的位置
    Src_in_world = [-4.667 ,450 ,-30.]
    # xyz_Dtctr 矩形探测器的原点在世界坐标系的位置
    Dtctr_in_world = [-4.667 ,-150 ,-2]
    # xvec_Dtctr 矩形探测器平面x轴在世界坐标系的单位向量 
    xvec_in_world = [1 , 0, 0]
    # yvec_Dtctr 矩形探测器平面y轴在世界坐标系的单位向量 
    yvec_in_world = [0 , 0,-1]
    # Sz_Dtctr 矩形探测器的物理尺寸，单位mm，(列，行)IDL习惯
    Sz_D = [500, 950]

    Img_size = n_col * n_row
    n_Verts_points = n_Verts * 3
    n_TRI_points = n_TRI * 3
    # print('Img_size = ',Img_size)
    te_verts = n_Verts
    te_verts_points = te_verts * 3
    te_tri = n_TRI
    te_tri_points = te_tri *3
    INPUT1 = c_float * te_verts_points
    INPUT2 = c_float * te_tri_points
    INPUT3 = c_float * te_verts
    INPUT4 = c_float * 3
    INPUT5 = c_float * 3
    INPUT6 = c_float * 3
    INPUT7 = c_float * 3 
    INPUT8 = c_float * 3
    INPUT9 = c_float * 3
    INPUT10 = c_float * 2
    INPUT11 = c_float * 2
    INPUT12 = c_float * Img_size
    INPUT13 = c_float * 1
    INPUT14 = c_float * 1
    INPUT15 = c_float * 1
    INPUT16 = c_float * te_verts
    INPUT17 = c_float * te_verts

    Verts = INPUT1()
    TRI = INPUT2()
    Labs_Verts = INPUT3() 
    Label_Fill = INPUT4() 
    Intensity_Fill = INPUT5() 
    Src = INPUT6() 
    xyz_Dtctr = INPUT7() 
    xvec_Dtctr = INPUT8() 
    yvec_Dtctr = INPUT9() 
    Sz_Dtctr = INPUT10() 
    Dim_Dtctr = INPUT11() 
    Img_Prj = INPUT12() 
    num_Verts = INPUT13()
    num_TRI = INPUT14()
    step = INPUT15()
    X_Vert = INPUT16()  
    Y_Vert = INPUT17()


    # Verts & Labs_Verts
    for i in range(te_verts):
        Verts[i*3] = Verts_point[i][0]
        Verts[i*3 + 1] = Verts_point[i][1]
        Verts[i*3 + 2] = Verts_point[i][2]
        X_Vert[i] = 0.
        Y_Vert[i] = 0.
    # Labs_Verts
    # Lungs的索引从0-7112，skeleton的索引从7113-255438，skin的索引从255439-257517
    for i in range(te_verts):
        if i <= 7112:
            Labs_Verts[i] = 1 #skin的label是1
        elif 7113 <= i <= 255438:
            Labs_Verts[i] = 2 #skeleton的label是2
        elif 255439 <= i <= 257517:
            Labs_Verts[i] = 3 #Lungs的label是3
    # TRI
    for i in range(te_tri):
        TRI[i*3] = TRI_point[i][0]    
        TRI[i*3 + 1] = TRI_point[i][1]   
        TRI[i*3 + 2] = TRI_point[i][2]    
    #print("len of Verts",len(Verts),"\nlen of TRI：",len(TRI),"\nlen of Labs_Verts:",len(Labs_Verts))
    # Label_Fill 可以被积分的组织的标签[1,2,3]
    for i in range(3):
        Label_Fill[i] = i + 1
        #Label_Fill[0] = 1
    # Intensity_Fill 组织的密度，用的是软组织、骨、肺部的CT值
    Intensity_Fill[0] = 200.
    Intensity_Fill[1] = 1000.
    Intensity_Fill[2] = -500.


    for ind,val in enumerate(Src_in_world):
        Src[ind] = val

    for ind,val in enumerate(Dtctr_in_world):
        xyz_Dtctr[ind] = val

    for ind,val in enumerate(xvec_in_world):
        xvec_Dtctr[ind] = val

    for ind,val in enumerate(yvec_in_world):
        yvec_Dtctr[ind] = val

    for ind,val in enumerate(Sz_D):
        Sz_Dtctr[ind] = c_float(val)
    # Dim_Dtctr 矩形探测器在世界坐标系中的尺寸
    Dim_D = [n_col,n_row]
    for ind,val in enumerate(Dim_D):
        Dim_Dtctr[ind] = c_float(val)
    # Img_Prj 输出图像即数字人模拟X光结果图,与探测器在世界坐标系中尺寸一致
    for i in range(Img_size):    
        Img_Prj[i] = 0.
    #print(Img_Prj)
    #print("len of Label_Fill:",len(Label_Fill),"\nlen of Intensity_Fill:",len(Intensity_Fill),"\nlen of Src:",len(Src))
    #print("len of xyz_Dtctr:",len(xyz_Dtctr),"\nlen of xvec_Dtctr:",len(xvec_Dtctr),"\nlen of yvec_Dtctr:",len(yvec_Dtctr))
    #print("len of Sz_Dtctr:",len(Sz_Dtctr),"\nlen of Dim_Dtctr:",len(Dim_Dtctr),"\nlen of Img_Prj:",len(Img_Prj))
    # num_Verts 总顶点数
    num_Verts[0] = c_float(te_verts)
    # num_TRI 总三角面片数
    num_TRI[0] = c_float(te_tri)
    # step 模拟X光积分的步长，单位mm
    step[0] = c_float(0.3)


    #-----------------------------3.使用 CDLL 导入 dll 文件，函数参数为 dll 路径--------------
    start = time.time()
    dll = CDLL(r"E:\\AAAA\\AAAA\\SeniorYear\\digital_human_Xray\\Cpp\\Dll1\\x64\\Debug\\Dll1.dll")
    Func = dll.ConePrj_Mesh_Natural
    Func.restype = c_float
    res = Func(Verts, TRI, Labs_Verts,Label_Fill , Intensity_Fill,
        Src, xyz_Dtctr, xvec_Dtctr, yvec_Dtctr, Sz_Dtctr,
        Dim_Dtctr, num_Verts, num_TRI, step, Img_Prj, X_Vert, Y_Vert)

    # 输出的：投影矩阵、模型所有三维点对应的二维投影点在矩形平面内的二维坐标
    out_array = np.zeros(Img_size)
    out_XVe = np.zeros(te_verts)
    out_YVe = np.zeros(te_verts)
    
    for i in range(Img_size):
        out_array[i] = Img_Prj[i]
    for i in range(te_verts):
        out_XVe[i] = X_Vert[i]
        out_YVe[i] = Y_Vert[i]

    # print(out_XVe,'\nout_XVe.shape=', out_XVe.shape)
    # print(out_YVe,'\nout_YVe.shape=', out_YVe.shape)
    end = time.time()
    print ('ConePrj total time =',end-start,'seconds!')
    # 一维转二维
    out_array = out_array.ravel('C')#先将数组变成C格式，为一维数组结果
    out_array = np.reshape(out_array, [n_col, n_row], order='F') #Dim:对应的二维维度，维度变换
    # print(out_array,'\nout_array.shape=', out_array.shape)
    # 去除矩阵中的nan值
    out_array = np.nan_to_num(out_array)
    out_array = out_array.T
    
    # 获取投影点在二维坐标系下对应的坐标
    
    np.savetxt(r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\elastix_transformix_file/SSM1_points.txt', out_XVe, delimiter=' ', fmt='%.02f')
    with open(r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\elastix_transformix_file/SSM1_points.txt', 'w') as f1:  # step1 & step4
        f1.write('point\n')
        f1.write(str(out_XVe.size))
        f1.writelines('\n')
        list1 = []
        list2 = []        
        for i in range(out_XVe.size):             
            x = out_XVe[i]
            y = out_YVe[i]
            list1.append(x)
            list2.append(y)           

            f1.writelines(str(x))
            f1.writelines(' ')
            f1.writelines(str(y))
            f1.writelines('\n')
    f1.close()   
    X = np.array(list1)
    Y = np.array(list2)

    # 裁剪X光图像，尽量与临床X光一致
    out_array_total = out_array.copy()

    point_2D = []
    for i in range(X.size):
        temp = []
        temp.append(X[i])
        temp.append(Y[i])
        point_2D.append(temp)
    point_2D = np.array(point_2D)
    # print(point_2D)
   




    # numpy to sitk 将投影矩阵保存为nii文件，便于Elastix配准
    save_path = r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/ConePrj_Mesh_Data/XConepSSM_test.nii'
    save_path2 = r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/ConePrj_Mesh_Data/XConepSSM_testSetwindow.nii'
    sitk_img = sitk.GetImageFromArray(out_array)
    sitk_img.SetSpacing((1.0, 1.0))
    sitk_img.SetOrigin((0., 0.))
    sitk.WriteImage(sitk_img, save_path)

    img2 = setWindowLevel(sitk_img, 2.72e4, 8.77e4)
    img2.SetSpacing((1.0, 1.0))
    img2.SetOrigin((0., 0.))
    array2 = sitk.GetArrayFromImage(img2)   
    sitk.WriteImage(img2, save_path2)
    
    # plt.figure('out_Img')
    # plt.imshow(out_array ,'gray')
    # plt.scatter(X, Y, s=0.5, c='r')


    

    # plt.figure('SetWindow_Img')
    # plt.imshow(array2 ,'gray')
  #  plt.show(block=True)

    return X, Y, out_array_total

# ---------------test-------------
if __name__ == "__main__":
    SSM_file = r'E:\AAAA\AAAA\Science\Scoliosis\Model/Mesh_Skin2079_Skel248326_Lungs7113.obj'
    # 逐行获取Verts和TRI
    with open(SSM_file) as file:
        verts = []
        tri = []    
        while 1:
            line = file.readline()   
            if not line:
                break
            strs = line.split()
            if strs[0] == 'v':
                verts.append((float(strs[1]),float(strs[2]),float(strs[3])))            
            if strs[0] == 'f':
                tri.append((float(strs[1])-1.,float(strs[2])-1.,float(strs[3])-1.))            
  
    Verts_point = np.array(verts)
    TRI_point = np.array(tri)
    ConePrj_Mesh(Verts_point, TRI_point)