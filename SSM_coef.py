from webbrowser import BackgroundBrowser
from cv2 import mean
import numpy as np
from TPS_RPM import warp_TPS
from SkinTotal_ICP_v2_update import Calc_SSMTsfm_2PtSets
from Test_ICP_v2_update import Test_Calc_SSMTsfm_2PtSets
from ConePrj_Mesh import ConePrj_Mesh
import pyvista as pv
import time
import os
import struct
from numpy import linalg as la
from elastix_f import elastix_f
from elastix_f import transformix_f
import codecs
import math
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sklearn.metrics as skm
from sklearn.neighbors import NearestNeighbors
from numba import jit
# def read_Eigenvect(file_path):
#     A = []
#     size = int(os.path.getsize(file_path) / 4) #float类型占用4字节
#     with open(file_path, 'rb') as fAAM:
#         for i in range(size):
#             data = fAAM.read(4)  #每次输出四个字节
#             data_float = struct.unpack('f', data)[0]
#             A.append(data_float)
#     A = np.array(A)
#     A = np.reshape(A,(187800, 33))
#     return A

    # fix_img = sitk.ReadImage(r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/ConePrj_Mesh_Data/XConepSSM_testSetwindow.nii')
    # result_img = sitk.ReadImage(r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\SSM_result/result.1.nii')
    # fix_array = sitk.GetArrayFromImage(fix_img)
    # result_array = sitk.GetArrayFromImage(result_img)

    # fix_array111 = (fix_array - np.min(fix_array)) / (np.max(fix_array) - np.min(fix_array))
    # result_array111 = (result_array - np.min(result_array)) / (np.max(result_array) - np.min(result_array))

    # ssd = np.sqrt(np.sum(np.square(result_array111 - fix_array111)))

def GetPtsFromSSM(file_path, type='SKIN'):
# 逐行获取Verts和TRI
    with open(file_path) as file:
        verts = []
        tri = []    
        py_tri = []
        while 1:
            line = file.readline()   
            if not line:
                break
            strs = line.split()
            if strs[0] == 'BONES': #-----仅取出体表顶点                
                if type == 'SKIN':
                    break
            if strs[0] == 'v':
                verts.append((float(strs[1]),float(strs[2]),float(strs[3])))            
            if strs[0] == 'f':
                tri.append((float(strs[1])-1.,float(strs[2])-1.,float(strs[3])-1.))  
                py_tri.append((3, float(strs[1]),float(strs[2]),float(strs[3])))        
    Verts = np.array(verts, dtype='float64')
                #SSM_verts = SSM_verts * 10000
    TRI = np.array(tri, dtype='float64')
    Py_TRI = np.array(py_tri, dtype='float64')
    return Verts, TRI, Py_TRI

def SSM_TPS(Suf_verts,target_file):
    mesh_target = pv.read(target_file)
    #-----降采样加快TPS速度
    Suf_verts_down = Suf_verts[:: 5] 
    mesh_target = mesh_target.decimate_boundary(target_reduction=1-(2000/mesh_target.n_points))
    target_verts = np.array(mesh_target.points, dtype='float64')
    #-----
    x, Suf_verts_TPS = warp_TPS(Suf_verts_down, Suf_verts, target_verts) #x是降采样后TPS结果，result是降采样前TPS结果
    
    # print(x.shape)
    # print(Suf_verts_TPS.shape)
    # print(x)
    # print(Suf_verts_TPS)
    #-----查看TPS配准效果
    # p = pv.Plotter()
    # p.add_mesh(SSM_vertsSuf, color="Crimson", show_edges=True, opacity=0.3)
    # p.add_mesh(target_verts, color="mintcream", show_edges=True, opacity=1)
    # p.add_mesh(x, color="green", show_edges=True, opacity=1)
    # p.add_mesh(Suf_verts_TPS, color="blue", show_edges=True, opacity=0.7)
    # p.show()

    return Suf_verts_TPS

def save_points(file_path, out_XVe, out_YVe):
    np.savetxt(file_path, out_XVe, delimiter=' ', fmt='%.02f')
    with open(file_path, 'w') as f1:  # step1 & step4
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
    return 

# 读入SSM的变形特征向量
def read_Eigenvect(file_path):
    A = []
    A = np.fromfile(file_path, dtype='float32') 
    A = np.array(A)
    A = np.reshape(A,(33, 187800))
    return A


# 计算归一化后X光的SSD误差平方和
def SSD_2DRegis(fix_file, result_file):
    fix_img = sitk.ReadImage(fix_file)
    result_img = sitk.ReadImage(result_file)
    fix_array = sitk.GetArrayFromImage(fix_img)
    result_array = sitk.GetArrayFromImage(result_img)

    fix_array111 = (fix_array - np.min(fix_array)) / (np.max(fix_array) - np.min(fix_array))
    result_array111 = (result_array - np.min(result_array)) / (np.max(result_array) - np.min(result_array))

    ssd = np.sqrt(np.sum(np.square(result_array111 - fix_array111)))

    return ssd

# 计算模型体表顶点与患者体表扫描最近点的距离
@jit(nopython=True)
def Calcu_close_dist(patient_points, SSM_points):
    result = 0    
    for point in patient_points:        
        point = np.expand_dims(point, axis =0)
        result = result + np.min(np.sqrt(np.sum(np.square(point - SSM_points), axis=1)))
                 

    return result / patient_points.shape[0]


# 读入变形后的二维模拟X光图像点坐标
def Get2DPoint_1From2DTrans(Points_file):
    
    f = codecs.open(Points_file, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
    line = f.readline()   # 以行的形式进行读取文件
    a = []    
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

if __name__ =='__main__':

    
    # 读入模型和患者体表
    #---------获取SSM模型体表顶点及连接关系-----------------------
    SSM_file = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\SSM_file/MeanMesh_Skin4132_Skel53610_Lungs4858.obj'
    mesh_SSM = pv.read(SSM_file)

    SSM_vertsSuf, SSM_triSuf,Py_TRISuf = GetPtsFromSSM(SSM_file)
    SSM_vertstal, SSM_trital, Py_TRITal = GetPtsFromSSM(SSM_file, type='All')
    #print('SSM_trital.shape = ', SSM_trital.shape)

    patient_file = r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/TPSRPM_Data/biantibiao.stl'
    mesh_patient = pv.read(patient_file)
    mesh_patient = mesh_patient.decimate_boundary(target_reduction=1-(1500/mesh_patient.n_points))
    verts_patient = np.array(mesh_patient.points, dtype='float64')

    # 输入参数设置
    TPS_input1 = SSM_vertsSuf.copy()
    TPS_input2 = patient_file

    SSM_vertstal_input = SSM_vertstal.copy()
                # a = pv.read(SSM_file)
                # b = pv.read(SSM_file)
                # a.points = RegistrationModel1['Vx']
                # b.points = SSM_vertstal

                # p = pv.Plotter()                
                # p.add_mesh(a, color="green", show_edges=True, opacity=1)                  
                # p.add_mesh(b, color="blue", show_edges=True, opacity=1)  
                # p.show()
    # 迭代配准流程
    start = time.time() 
    j = 0
    trial = 5
    mean_d = 2
    thred = 0.2
    while(j < trial and mean_d > thred):
        j += 1           
        print('Circulate',j, 'Running!')
        #-----1.SSM体表与病人体表做TPSRPM-----------
        SSM_verts_TPS = SSM_TPS(TPS_input1, TPS_input2)

        #---------------------------------------TPS变形后的体表顶点与变形向量求出系数，进行第一次统计形状模型变形------------------------------
        Vy1 = SSM_verts_TPS.copy() #求系数函数的输入
        Verts_mean = TPS_input1.flatten()
        Verts_mean_total = SSM_vertstal_input.copy().flatten()
        # print('Vy,shape=', Vy1.shape)
        #-----读入变形向量，并取出包含SKIN的值
        Eigvect_file = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\SSM_file/PCs_SkinSkelLungs187800x33.bin'
        Eigenvect = read_Eigenvect(Eigvect_file)
        Eigenvect1 = Eigenvect.copy()
        Eigenvect_SKIN = Eigenvect1[:, :4132 * 3] #只用SKIN部分
        Eigenvect_total = Eigenvect1
        #print(Eigenvect_SKIN, Eigenvect_SKIN.shape)

        num_Pts1 = Verts_mean.shape[0]
        numPC1 = Eigenvect_SKIN.shape[0]
        #-----构建求系数函数的输入SSM模型
        # SSM1 = {'Verts_mean': Verts_mean, 'Conn_mean': SSM_trital,
        #     'Eigenvect': Eigenvect_SKIN, 'numPC': numPC1, 
        #     'num_Verts': num_Pts1, 'Verts_mean_total': Verts_mean_total, 
        #     'Eigenvect_total': Eigenvect_total}
        SSM1 = {'Verts_mean_Skin': Verts_mean, 'Eigenvect_Skin': Eigenvect_SKIN, 
             'numPC': numPC1, 'Verts_mean_total': Verts_mean_total, 
             'Eigenvect_total': Eigenvect_total}

        RegistrationModel1 = Calc_SSMTsfm_2PtSets(Vy1,SSM1)
        #RegistrationModel1_test = Test_Calc_SSMTsfm_2PtSets(Vy1,SSM1)



        mesh_SSM1 = mesh_SSM.copy()        
        mesh_SSM.points = RegistrationModel1['Vx'].copy()
        p = pv.Plotter()
        p.background_color = 'w'
        p.add_mesh(mesh_SSM, color="Crimson", show_edges=True, opacity=1)
        p.add_mesh(mesh_patient, color="mintcream", show_edges=True, opacity=0.3)
        p.add_mesh(SSM_verts_TPS, color="green", show_edges=True, opacity=0.3)
        p.show()
        # a = RegistrationModel1_test['Vx']
        # b = RegistrationModel1['Vx'][:4132]
########------------------观察PC叠加----------

            
        # mesh_SSM.points = RegistrationModel1['Vx'].copy()
        # numPC = 20
        # ind_PCs =  np.arange(numPC)
        # c = RegistrationModel1['coef']
        # Vx0 = SSM1['Verts_mean_total'] + np.dot(c[ind_PCs], SSM1['Eigenvect_total'][ind_PCs,:])
        
            # mesh_SSM1 = mesh_SSM.copy() 
            # mesh_SSM.points = RegistrationModel1['Vx']

            # p = pv.Plotter()
            # p.background_color = 'w'
            # p.add_mesh(mesh_SSM, color="Crimson", show_edges=True, opacity=0.5, 
            #            style='wireframe', lighting=False)     
            # p.add_mesh(SSM_verts_TPS, color="green", show_edges=True, opacity=0.5, 
            #            style='wireframe', lighting=False)       
            
            # p.show()
########-------------------------------------------------

        # p = pv.Plotter()
        # p.background_color = 'w'
        # p.add_mesh(mesh_SSM, color="Crimson", show_edges=True, opacity=1)
     
        # p.add_mesh(mesh_SSM1, color="green", show_edges=True, opacity=1)
        
        # p.show()
        #p.show_grid()
        #---------------------------------------对一次变形后的SSM模型做模拟X光投影----------------------------------
        # 进行空间中旋转、平移变换，使SSM模型姿态符合预设投影的参数
        SSM1_verts = RegistrationModel1['Vx'].copy()
        # SSM1_verts[:, 2] = SSM1_verts[:, 2] + 335
        # SSM1_verts[:, 1] = -SSM1_verts[:, 1]
        # SSM1_verts = SSM1_verts[:, [0, 2, 1]]

        #SSM0 = pv.PolyData(RegistrationModel['Vx'], SSM_trital)
        # 保存成obj文件
        #pv.save_meshio(r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\SSM_trans_Data/SSM_trans1.obj', SSM0)
        #pv.save_meshio(r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\SSM_trans_Data/SSM_Conprjtttt.obj', mesh_SSM1)
        # Src x射线源在世界坐标系的位置
        Src_in_world = [-4.667, 450, -30]
        # xyz_Dtctr 矩形探测器的原点在世界坐标系的位置
        Dtctr_in_world = [-4.667, -150, -2]
        # xvec_Dtctr 矩形探测器平面x轴在世界坐标系的单位向量 
        xvec_in_world = [1, 0, 0]
        # yvec_Dtctr 矩形探测器平面y轴在世界坐标系的单位向量 
        yvec_in_world = [0, 0, -1]
        Src_in_world = np.array(Src_in_world)
        Dtctr_in_world = np.array(Dtctr_in_world)
        xvec_in_world = np.array(xvec_in_world)        
        yvec_in_world = np.array(yvec_in_world)

        X0_2Dpoint, Y0_2Dpoint, ConePrj_img = ConePrj_Mesh(SSM1_verts, SSM_trital, Dtctr_in_world, 
                                                           xvec_in_world, yvec_in_world, Src_in_world)
        print('ok1')
        #---------------------------------------对一次变形后的SSM模型的模拟X光投影做2D配准,并获得变形后的二维点坐标----------------------------------
        i_mov = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\pretreatment\patients/BYM_Downsample.nii'
        i_fix = r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/ConePrj_Mesh_Data/XConepSSM_testSetwindow.nii'
        elas_out_path = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\SSM_result'
        elastix_f(i_fix, i_mov, elas_out_path)
        print('ok2')
        point0_file = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\elastix_transformix_file/SSM1_points.txt'
        transf_output = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\transformix_result'
        transf_parameter0 = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\SSM_result/TransformParameters.0.txt'
        transf_parameter1 = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\SSM_result/TransformParameters.1.txt'
        transformix_f(point0_file, transf_output, transf_parameter1)
        print('ok3')
        result_file = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\SSM_result/result.1.nii'
        fix_img = sitk.ReadImage(i_fix)
        result_img = sitk.ReadImage(r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\SSM_result/result.1.nii')
        fix_array = sitk.GetArrayFromImage(fix_img)
        result_array = sitk.GetArrayFromImage(result_img)
        x = np.reshape(fix_array, -1)
        y = np.reshape(result_array, -1)
        hxx = skm.mutual_info_score(x, y)
        print('\nCirculate', j, '互信息=', hxx)

        # X0_2Dpoint, Y0_2Dpoint = Get2DPoint_1From2DTrans(Points_tran_file)

        # point1_file = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\elastix_transformix_file/SSM2_points.txt'
        # save_points(point1_file, X0_2Dpoint, Y0_2Dpoint)
        # transformix_f(point1_file, transf_output, transf_parameter1)
        
        # 验证变形后投影点是否正确
        Points_tran_file = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\transformix_result/outputpoints.txt'
        X1_2Dpoint, Y1_2Dpoint = Get2DPoint_1From2DTrans(Points_tran_file)
        

        img_2Dtrans1_path = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\pretreatment\patients/BYM_Downsample.nii'
        img_2Dt1 = sitk.ReadImage(img_2Dtrans1_path)

        img_2Dt1_array = sitk.GetArrayFromImage(img_2Dt1)

        # plt.figure('out_Img')
        # plt.imshow(img_2Dt1_array ,'gray')
        # plt.scatter(X1_2Dpoint, Y1_2Dpoint, s=0.5, c='r')
        # plt.show(block=True)
        #---------------------------------------读入配准前后的二维点坐标，转换至world坐标系下三维坐标----------------------------------

        # 配准前2D点的三维坐标
        SSM1_2D0_in_world = []
        for i in range(X0_2Dpoint.size):
            a = []    
            a = Dtctr_in_world + X0_2Dpoint[i] * xvec_in_world +Y0_2Dpoint[i] * yvec_in_world  #矩形平面原点坐标+x*x轴向+y*y轴向
            SSM1_2D0_in_world.append(a)
        SSM1_2D0_in_world = np.array(SSM1_2D0_in_world)
        # 配准后2D点的三维坐标
        SSM1_2D1_in_world = []
        for i in range(X1_2Dpoint.size):
            a = []    
            a = Dtctr_in_world + X1_2Dpoint[i] * xvec_in_world +Y1_2Dpoint[i] * yvec_in_world  #矩形平面原点坐标+x*x轴向+y*y轴向
            SSM1_2D1_in_world.append(a)
        SSM1_2D1_in_world = np.array(SSM1_2D1_in_world)
        print('ok4')
        # print('SSM1_2D0_in_world.shape:', SSM1_2D0_in_world.shape,'\n', SSM1_2D0_in_world)
        # print('SSM1_2D1_in_world.shape:', SSM1_2D1_in_world.shape,'\n', SSM1_2D1_in_world)

        #---------------------------------------利用模拟三角形投影原理，进行反投影，获得变形后模型三维点----------------------------------
        SSM2_verts = []
        O = Src_in_world
        for i in range(SSM1_2D0_in_world[:,0].size):
            temp = []
            M = O - SSM1_2D1_in_world[i]
            m = math.sqrt(M[0] * M[0] + M[1] * M[1] + M[2] * M[2])
            N = O - SSM1_verts[i]
            n = math.sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2])
            P = SSM1_2D1_in_world[i] - SSM1_2D0_in_world[i]
            P1 = P / (math.sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]))
            p = math.sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2])
            q = n * p / m
            temp.append(SSM1_verts[i] + q * P1)
            SSM2_verts.append(temp)
        SSM2_verts = np.squeeze(np.array(SSM2_verts))

        # 验证反投影
        # p = pv.Plotter()
        # #p.add_mesh(SSM1_verts, color="blue", show_edges=True, opacity=1)
        # p.add_mesh(SSM2_verts, color="Crimson", show_edges=True, opacity=1)
        # p.show()

        #---------------------------------------利用2D/3D配准后的统计形状模型顶点求形状系数，进行第二次统计形状模型变形----------------------------------
        Vy2 = SSM2_verts.copy() #求系数函数的输入
        Verts_mean = SSM1_verts.flatten()
        Verts_mean_total2 = SSM1_verts.flatten()
        num_Pts2 = Verts_mean.shape[0]
        # 这次求形状系数需要使用对应全部顶点的特征向量
        
        Eigenvect2 = Eigenvect.copy()        
        Eigenvect_total2 = Eigenvect2
        numPC2 = Eigenvect_total2.shape[0]
        #-----构建求系数函数的输入SSM模型
        # SSM2 = {'Verts_mean': Verts_mean, 'Conn_mean': SSM_trital,
        #     'Eigenvect': Eigenvect, 'numPC': numPC2, 
        #     'num_Verts': num_Pts2, 'Verts_mean_total': Verts_mean_total, 
        #     'Eigenvect_total': Eigenvect
        #     }
        SSM2 = {'Verts_mean_Skin': Verts_mean, 'Eigenvect_Skin': Eigenvect_total2, 
             'numPC': numPC2, 'Verts_mean_total': Verts_mean_total2, 
             'Eigenvect_total': Eigenvect_total2}        
        RegistrationModel2 = Calc_SSMTsfm_2PtSets(Vy2,SSM2)

        # 判断相邻两次循环的变形幅度（所有顶点在前后两次迭代的位置距离的平均值）
        if j == 1:
            last_Verts = SSM1_verts.copy()
            This_Verts = RegistrationModel2['Vx'].copy()
        else:
            last_Verts = This_Verts.copy()
            This_Verts = RegistrationModel2['Vx'].copy()

        diffe = This_Verts - last_Verts
        d_sum = []
        for i in range(This_Verts.shape[0]):
            m = This_Verts[i]
            n = last_Verts[i]
            d_point = np.sqrt((m[0] - n[0])**2 + (m[1] - n[1])**2 + (m[2] - n[2])**2)
            d_sum.append(d_point)
        mean_dist =np.mean(d_sum)
        # ttt = RegistrationModel2['Vx'][0:4132].copy()
        # ttt = np.array(ttt, np.float64) 
        # ppp = np.array(mesh_patient.points.copy(),np.float64)
        # ddd = Calcu_close_dist(mesh_patient.points, ttt)
        # print('Circulate',j,'distance = ', ddd)
        print('Circulate',j, ' mean_d = ', mean_dist)
        # 设置新一轮循环的输入
        TPS_input1 = RegistrationModel2['Vx'][0:4132].copy()
        SSM_vertstal_input = RegistrationModel2['Vx'].copy()
        print('Circulate',j, 'Complete!\n')
###################################
        # dist = []
        # d_sample_patient =mesh_patient.points[::200]
        # a = d_sample_patient.copy()
        # b = RegistrationModel2['Vx'][0:4132].copy()
        # for i in range(a.shape[0]):   
        #     if i >20:
        #         break     
        #     temp = []
        #     
        #     for j in range(b.shape[0]):        #         
        #         m = a[i]
        #         n = b[j]
        #         d_p = np.sqrt((m[0] - n[0])**2 + (m[1] - n[1])**2 + (m[2] - n[2])**2) 
        #         temp.append(d_p)
        #     dist.append(np.min(temp))


    end = time.time()
    total_time = end - start
    print(time.strftime("%H:%M:%S", time.gmtime(total_time)))

    print('1231213213212312231')


    ###################################
    cc = 11
        #  mesh_SSM1 = mesh_SSM.copy()
        #  mesh_SSM2 = mesh_SSM.copy()
        #  mesh_SSM1.points = SSM2_verts
        #  mesh_SSM2.points = RegistrationModel2['Vx']
        #  p = pv.Plotter()
        #  p.background_color = 'w' 
         
        #  p.add_mesh(mesh_SSM2, color="red", style='wireframe',lighting=False, show_edges=True, opacity=0.5)
        #  p.add_mesh(mesh_SSM1, color="green", style='wireframe',lighting=False, show_edges=True, opacity=0.5)
        #  p.show()
######################################

        # p = pv.Plotter()
        # p.background_color = 'w' 
        # p.add_mesh(RegistrationModel2['Vx'], color="Crimson", lighting=False, show_edges=True, opacity=1)
        # p.add_mesh(mesh_SSM1, color="mintcream", show_edges=True, lighting=False, opacity=1)
        # p.add_mesh(SSM1_verts, color="green", show_edges=True, lighting=False, opacity=1)
        # p.show()

        # p = pv.Plotter()    
        # p.background_color = 'w'    
        # p.add_mesh(mesh_SSM, color="green", show_edges=True, style='wireframe', lighting=False, opacity=0.5)
        # p.add_mesh(mesh_SSM1, color="Crimson", show_edges=True, style='wireframe', lighting=False, opacity=0.5)
        # p.show()