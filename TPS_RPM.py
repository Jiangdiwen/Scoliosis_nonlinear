"""
PURPOSE:
  1. TPS-RPM
INPUTS: NA
OUTPUT: NA
AUTHOR: XIAOXIAN JIN
CREATE: 2021-10-28
"""

import numpy as np
import math
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
import vtk

def warp_TPS(SSM0, SSM1, target):
    # tps_rpm计算变换矩阵
    Vx, V0, d, w, scale_X, center_X, scale_V0, center_V0 = TPS_RPM(SSM0, target)  # V0是SSM0已经归一化

    K = V0.shape[0]
    N = SSM1.shape[0]
    Dim = SSM1.shape[1]
    #print('1', SSM1)
    # 归一化
    SSM1 = (SSM1 - center_V0) / scale_V0  # SSM1与SSM0进行相同的归一化
    #print('2', SSM1)
    # 非线性变换参数
    V0 = np.concatenate((np.ones((K, 1)), V0), axis=1)
    SSM1 = np.concatenate((np.ones((N, 1)), SSM1), axis=1)
    PHI = ComputePHI(SSM1, V0)
    #print('3', SSM1)
    # 变形后
    SSM1 = np.dot(SSM1, d) + np.dot(PHI, w)
    #print('d:', d, '\nPHI:', PHI, '\nw:', w)
    #print('4', SSM1)
    SSM1 = SSM1[:, 1: Dim + 1]
    #print('5', SSM1) 
    # 去归一化
    SSM1 = SSM1 * scale_X + center_X  # 去归一化还原到目标尺寸
    #print('6', SSM1)
    return Vx, SSM1

def TPS_RPM(V0, X, anneal_rate=0.5, T_init=0.05, T_fac=500, lambda1=1):
    """
    V0: 待配准点云
    X: 目标点云
    anneal_rate: 退火速率
    T_init: 初始温度
    T_fac: T_final=T_init/T_fac

    """
    print('TPS_RPM is Running！')
    Iter_perT = 5  # 对每个T的最大循环次数
    lambda1_init = lambda1  # 计算d w
    lambda2_init = 0.01

    K = V0.shape[0]  # 待配准点云 K
    N = X.shape[0]  # 目标点云 N
    Dim = V0.shape[1]

    # 归一化
    center_V0 = (sum(V0)/K).reshape((1, Dim))
    V0 = V0 - np.dot(np.ones((K, 1)), center_V0)
    scale_V0 = np.sqrt((sum(np.power(V0, 2))/K).reshape((1, Dim)))
    V0 = V0 / np.dot(np.ones((K, 1)), scale_V0)

    center_X = (sum(X) / N).reshape((1, Dim))
    X = X - np.dot(np.ones((N, 1)), center_X)
    scale_X = np.sqrt((sum(np.power(X, 2)) / N).reshape((1, Dim)))
    X = X / np.dot(np.ones((N, 1)), scale_X)

    # 初始化权重m
    m = np.ones((K, N)) / (K*N)
    T0 = math.pow(max(V0[:, 0]), 2)  # pow求平方
    moutlier = 1 / math.sqrt(T0) * math.exp(-1)
    m_outliers_row = np.ones((1, N)) * moutlier
    m_outliers_col = np.ones((K, 1)) * moutlier

    # 退火
    T = T_init
    T_final = T_init / T_fac

    Vx = V0  # Vx表示f(v),初始值与待配准点云一致
    z = np.concatenate((np.ones((K, 1)), V0), axis=1)  # V0增加一维，齐次？

    flag_stop = 0
    while flag_stop != 1:
        w = np.zeros((K, Dim+1), dtype='float64')
        d = np.zeros((Dim+1, Dim+1), dtype='float64')
        Vy = np.zeros((K, Dim), dtype='float64')
        for i in range(1, Iter_perT+1):  # 左闭右开区间 1,2,...,Iter_perT
            # nan
            Vx_last = Vx
            w_last = w
            d_last = d
            Vy_last = Vy

            # 更新权重矩阵m
            m = ComputeM(Vx, X, K, N, T, moutlier, m_outliers_row, m_outliers_col)

            # 更新变换矩阵
            Vy = np.dot(m, X) / np.dot(np.sum(m, axis=1).reshape((K, 1)), np.ones((1, Dim)))  # Vy 使用权重计算出的目标点云位置

            # nan
            ind_y = np.isnan(Vy)
            Vy[ind_y] = Vy_last[ind_y]
            
            lambda1 = lambda1_init * K * T
            lambda2 = lambda2_init * K * T

            w, d, PHI = ComputeT(V0, Vy, lambda1, lambda2)  # w:K*(D+1) d:(D+1)*(D+1) PHI:K*K
            if np.isnan(Vy).any():
                print('Vy=', Vy.shape, '\n', Dim, K, N)
            
            if np.isnan(w).any():
                print('w=', w)
            # nan
            #ind_w = np.isnan(w)
            #w[ind_w] = w_last[ind_w]
            #ind_d = np.isnan(d)
            #d[ind_d] = d_last[ind_d]

            #print('TPS_RPMzhong:', '\nw:', w, '\nd:', d)
            # 更新点云
            PHI = ComputePHI(z, z)
            Vx = np.dot(z, d) + np.dot(PHI, w)
            Vx = Vx[:, 1: Dim + 1]

            # nan
            ind_x = np.isnan(Vx)
            Vx[ind_x] = Vx_last[ind_x]

        T = T * anneal_rate

        if T < T_final:
            flag_stop = 1

    # surf1 = pv.PolyData(V0)
    # surf2 = pv.PolyData(X)
    # surf3 = pv.PolyData(Vx)
    # plt = pv.Plotter()
    # plt.set_background(color='white')
    # plt.add_mesh(surf1, color="Crimson", opacity=0.5, show_edges=True)  # 待配
    # plt.add_mesh(surf2, color="green", opacity=0.5, show_edges=True)  # 目标
    # plt.add_mesh(surf3, color="blue", opacity=0.5, show_edges=True)  # 结果
    # plt.show()

    # 恢复位置和尺度
    Vx = Vx * np.dot(np.ones((K, 1)), scale_X) + np.dot(np.ones((K, 1)), center_X)

    return Vx, V0, d, w, scale_X, center_X, scale_V0, center_V0

def ComputeM(Vx, X, K, N, T, moutlier, m_outliers_row, m_outliers_col):
    K = Vx.shape[0]  # 待配准点云 K
    N = X.shape[0]  # 目标点云 N
    Dim = Vx.shape[1]

    tmp = np.zeros((K, N))
    for it_dim in range(0, Dim):  # 左闭右开 0,1,2
        tmp = tmp + np.power(np.dot(Vx[:, it_dim:it_dim + 1], np.ones((1, N))) -
                             np.dot(np.ones((K, 1)), np.transpose(X[:, it_dim:it_dim + 1])), 2)
    m_tmp = 1 / math.sqrt(T) * np.exp(-tmp / T)
    # m_tmp = m_tmp + np.random.randn(K, N) * (1 / K) * 0.001

    m = m_tmp
    # normalize accross the outliers as well
    sy = sum(m) + m_outliers_row  # sum 按列求和
    m = m / np.dot(np.ones((K, 1)), sy)  # 权重矩阵m K*N

    # ---2021.10.12新增加 normalize1
    # sy = sum(m) + m_outliers_row  # sum 按列求和
    # m1 = m / np.dot(np.ones((K, 1)), sy)  # 权重矩阵m K*N
    # sx = sum(np.transpose(m)).reshape((K, 1)) + m_outliers_col  # sum 按行求和
    # m2 = m / np.dot(sx, np.ones((1, N)))  # 权重矩阵m K*N
    # m = (m1 + m2)/2

    # ---2021.10.15新增加 normalize2
    # moutlier = 1 / K * 0.1
    # m_outliers_row = np.ones((1, N)) * moutlier
    # m_outliers_col = np.ones((K, 1)) * moutlier
    # m = C_normalize_m(K, N, m, m_outliers_col, m_outliers_row)

    return m


def C_normalize_m (K, N, m, m_outliers_col, m_outliers_row):
    norm_threshold = 0.05
    norm_maxit = 10

    norm_it = 0
    flag = 0
    while flag == 0:
        sy = sum(m) + m_outliers_row  # sum 按列求和
        m = m / np.dot(np.ones((K, 1)), sy)  # 权重矩阵m K*N
        m_outliers_row = m_outliers_row / sy

        sx = sum(np.transpose(m)).reshape((K, 1)) + m_outliers_col  # sum 按行求和
        m = m / np.dot(sx, np.ones((1, N)))  # 权重矩阵m K*N
        m_outliers_col = m_outliers_col / sx

        pre_err = np.dot(np.transpose(sx - 1), (sx-1)) + np.dot((sy-1), np.transpose(sy - 1))
        err = (pre_err / (K + N))

        if err < np.power(norm_threshold, 2):
            flag = 1

        norm_it = norm_it + 1
        if norm_it >= norm_maxit:
            flag = 1
    return m


def ComputeT(V, Y, lambda1, lambda2):
    K = V.shape[0]  # V.shape=Y.shape
    D = V.shape[1]

    V = np.concatenate((np.ones((K, 1)), V), axis=1)
    Y = np.concatenate((np.ones((K, 1)), Y), axis=1)

    # 计算PHI
    """
    PHI = np.zeros(K, D)
    for it_dim in range(1, D+1):
        tmp = V[:, it_dim + 1]() * np.ones(1, D) - np.ones(K, 1) * np.transpose(V[:, it_dim + 1])
        tmp = tmp * tmp
        PHI = PHI + tmp
        PHI = - np.sqrt(PHI)
    """
    PHI = ComputePHI(V, V)

    # 计算QR
    q, r = np.linalg.qr(V, mode="complete")
    q1 = q[:, 0:D+1]  # K*(D+1)
    q2 = q[:, D+1:K]  # N*(K-D-1) K*(K-D-1)?
    R = r[0:D+1, 0:D+1]  # (D+1)*(D+1)

    # 计算w d
    pre_gamma = np.dot(np.dot(np.transpose(q2), PHI), q2) + lambda1 * np.eye(K-D-1)  # (K-D-1)*(K-D-1)
    gamma = np.dot(np.dot(np.linalg.inv(pre_gamma), np.transpose(q2)), Y)  # (K-D-1)*(D+1)
    w = np.dot(q2, gamma)  # K*(D+1)

    pre_A1 = np.dot(np.transpose(R), R) + lambda2 * np.eye(D+1)  # (D+1)*(D+1)
    pre_A2 = Y - np.dot(np.dot(PHI, q2), gamma)  # K*(D+1)
    pre_A3 = np.dot(np.dot(np.transpose(R), np.transpose(q1)), pre_A2) - np.dot(np.transpose(R), R)  # (D+1)*(D+1)
    A = np.dot(np.linalg.inv(pre_A1), pre_A3)  # (D+1)*(D+1)
    d = A + np.eye(D+1)  # (D+1)*(D+1)

    return w, d, PHI


def ComputePHI(V, Y):
    n = V.shape[0]
    m = Y.shape[0]
    D = V.shape[1]

    # V = np.concatenate(np.ones(n, 1), V)
    # Y = np.concatenate(np.ones(m, 1), Y)

    # 计算PHI
    PHI = np.zeros((n, m))
    for it_dim in range(1, D):  # 左闭右开 1，2,3
        tmp = np.dot(V[:, it_dim:it_dim + 1], np.ones((1, m))) - np.dot(np.ones((n, 1)), np.transpose(Y[:, it_dim:it_dim + 1]))
        tmp = tmp * tmp  # 每个元素计算平方
        PHI = PHI + tmp
    if D-1 == 3:
        PHI = - np.sqrt(PHI)
    elif D-1 == 2:
        mask = np.zeros((n, m))
        mask[PHI < 1e-10] = 1
        PHI = 0.5 * PHI * np.log(PHI + mask) * (1 - mask)
    return PHI


def Check(a, b):
    # temp_p = "C:/Users/USER/Desktop/点云配准/TPS-RPM/TPS-RPM-Matlab/check.mat"
    # import scipy.io as scio
    # mat = scio.loadmat(temp_p)
    # data = mat[a]
    # check = (data == b)
    check = (a == b)
    return check


def Readmat(a, b):
    temp_p = "C:/Users/USER/Desktop/点云配准/TPS-RPM/TPS-RPM-Matlab/" + a + ".mat"
    import scipy.io as scio
    mat = scio.loadmat(temp_p)
    data = mat[b]
    return data

def GetpointsfromObj(Obj_file):
    # 逐行获取Verts和TRI
    #start = time.time()
    with open(Obj_file) as file:
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
            
    #end = time.time()
    #print ('total time =',end-start,'seconds')
    Verts_points = np.array(verts,dtype='float64')
    TRI_points = np.array(tri)

    return Verts_points


# ---------------test-------------
if __name__ == "__main__":
    print('zenmehuishi')
   
    file_path = r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/TPSRPM_Data/'
    #path_V0 = file_path + 'chaiyihan.stl'
    path_V0 = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\TPSRPM_Data\Mean_Torso.obj'
    path_X = file_path + 'biantibiao.stl'
    # path_V0 = "C:/Users/USER/Desktop/body/meanSkin_femaleSSM.obj"  # 体表
    # path_X = "C:/Users/USER/Desktop/body/caixinyan.stl"
    
    V0 = GetpointsfromObj(path_V0)
    V0 = V0[:2291] #取模型的体表进行配准
    #V1 = V0 #不降采样体表，带入Warp_TPS查看是否能将二者位置对齐
    #V1 = V1[:,[0, 2, 1]]
    #V1[:,1] = -V1[:,1]
    # 降采样模型 加快算法速度 
    SSM0_verts = V0[: : 5] 
    SSM0_verts = SSM0_verts[:, [0, 2, 1]] #简单旋转
    SSM0_verts[:, 1] = -SSM0_verts[:, 1] #再旋转正反面
    SSM0_verts[:, 2] = SSM0_verts[:, 2] - 335
    mesh_X = pv.read(path_X)
    # pyvista读入整体模型
    SSM1 = pv.read(path_V0)
    SSM1.points = SSM1.points[:,[0, 2, 1]]
    SSM1.points[:,1] = -SSM1.points[:,1]
    SSM1.points[:, 2] = SSM1.points[:, 2] - 335
    SSM1_verts = SSM1.points

    mesh_X = mesh_X.decimate_boundary(target_reduction=1-(2500/mesh_X.n_points)) #降采样固定模型
    target_verts = np.array(mesh_X.points, dtype='float64')


    import time                                

    start = time.time()

    anneal_rate = 0.93  # 0.93  一般在[0.9,0.99]
    T_init = 0.5  # 0.5
    T_fac = 500
    lambda1 = 1
    print()
    # 调用TPS-RPM算法，并且将获得的TPS变换作用于原始点云
    result_SSM0, result_SSM1 = warp_TPS(SSM0_verts, SSM1_verts, target_verts)
    result = SSM1.copy()
    result.points = result_SSM1
    #-------------------在脚本中代入SSM文件试试
    SSM_file = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\SSM_file/MeanMesh_Skin4132_Skel53610_Lungs4858.obj'
    Verts = GetpointsfromObj(SSM_file)
    SSM_verts = Verts[:4132]
    
    SSM_verts_down = SSM_verts[:: 5] 

    #result_SSM0, result_SSM1 = warp_TPS(SSM_verts_down, SSM1_verts, target_verts)

    end = time.time()
    total_time = end - start
    print(total_time)
    print(time.strftime("%H:%M:%S", time.gmtime(total_time)))
    
    
    
    surf_fix = pv.PolyData(mesh_X)
    surf_mov = pv.PolyData(result)
    # mesh_V0.plot(show_edges=True, color=True)
    # mesh_X.plot(show_edges=True, color=True)
    print('draw beginning')
    p = pv.Plotter()
    #p.add_mesh(result_SSM0, color="red", show_edges=True, opacity=1)
    print('2')
    p.add_mesh(surf_fix, color="mintcream", show_edges=True, opacity=1)
    p.add_mesh(surf_mov, color="green", show_edges=True, opacity=0.5)
    #p.add_mesh(SSM1, color="blue", show_edges=True, opacity=0.3)
    p.show()
    print('save beginning')

    # 将配准后的点云进行旋转、平移，使坐标位置与最初一致，便于后续的模型X光投影
    result.points[:, 2] = result.points[:, 2] + 335
    result.points[:, 1] = -result.points[:, 1]
    result.points = result.points[:, [0, 2, 1]]
    # 保存成obj文件
    pv.save_meshio(r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/TPSRPM_Data/result/TPS_Torso.obj', result)
    #pv.save_meshio(r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/TPSRPM_Data/result/V0.stl', V0)
    #pv.save_meshio(r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/TPSRPM_Data/result/X.stl', mesh_X)


    input()








