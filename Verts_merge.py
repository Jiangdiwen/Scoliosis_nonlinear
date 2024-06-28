'''
Author: Yanxin_Jiang
Date: 2022-07-13 16:09:58
LastEditors: Yanxin_Jiang
LastEditTime: 2022-07-15 15:47:55
Description: 

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import pyvista as pv
import numpy as np
import time
from ConePrj_Mesh import ConePrj_Mesh
from TPS_RPM import warp_TPS

# 逐行获取Verts和TRI
def GetPtsFromObj(file_path):
    with open(file_path) as file:
        verts = []
        tri = []    
        py_tri = []
        while 1:
            line = file.readline()   
            if not line:
                break
            strs = line.split()
            if strs[0] == 'v':
                verts.append((float(strs[1]),float(strs[2]),float(strs[3])))            
            if strs[0] == 'f':
                tri.append((float(strs[1])-1.,float(strs[2])-1.,float(strs[3])-1.))  
                py_tri.append((3, float(strs[1]),float(strs[2]),float(strs[3])))        
    Verts = np.array(verts, dtype='float64')
                #SSM_verts = SSM_verts * 10000
    TRI = np.array(tri, dtype='float64')
    Py_TRI = np.array(py_tri, dtype='float64')
    return Verts, TRI

def SSM_TPS(Total_verts,target_file):
    mesh_target = pv.read(target_file)
    #----旋转模型
    Total_verts = Total_verts[:,[0,2,1]]
    Total_verts = Total_verts[:,[1,0,2]]
    Total_verts = Total_verts[:,[2,1,0]]
    Total_verts[:,0] = -Total_verts[:,0]
    Total_verts[:,2] = -Total_verts[:,2]
    #-----降采样加快TPS速度
    Suf_verts = Total_verts[:2079]
    Suf_verts_down = Suf_verts[:: 5]    


    mesh_target = mesh_target.decimate_boundary(target_reduction=1-(2000/mesh_target.n_points))
    target_verts = np.array(mesh_target.points, dtype='float64')
    #-----
    x, Total_verts = warp_TPS(Suf_verts_down, Total_verts, target_verts) #x是降采样后TPS结果，result是降采样前TPS结果
    
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

    return Total_verts


if __name__ == '__main__':

    #----------------------------将不同部位模型合并-----------------------------------------
    Skin_file = r'E:\AAAA\AAAA\A_Science\Scoliosis\Model/Skin2079_Ref.obj'
    Skel_file = r'E:\AAAA\AAAA\A_Science\Scoliosis\Model/Skel248326_Ref.obj'
    Luns_file = r'E:\AAAA\AAAA\A_Science\Scoliosis\Model/Lungs7113_Ref.obj'
    # 获取模型各部分顶点及连接关系
    Skin_verts, Skin_tri = GetPtsFromObj(Skin_file)
    Skel_verts, Skel_tri = GetPtsFromObj(Skel_file)
    Luns_verts, Luns_tri = GetPtsFromObj(Luns_file)
    # 合并
    model_verts = np.concatenate((Skin_verts, Skel_verts, Luns_verts), axis=0)
    model_tri = np.concatenate((Skin_tri, Skel_tri+2079, Luns_tri+2079+248326), axis=0)
    
    #--------------------------------模型体表与患者体表做TPS，获得的变换作用于整体模型-------------------------------
    Skin_file = r'E:\AAAA\AAAA\A_Science\Scoliosis\Model/Skin2079_Ref.obj'
    Skel_file = r'E:\AAAA\AAAA\A_Science\Scoliosis\Model/Skel248326_Ref.obj'
    Luns_file = r'E:\AAAA\AAAA\A_Science\Scoliosis\Model/Lungs7113_Ref.obj'
    Mode_file = r'E:\AAAA\AAAA\A_Science\Scoliosis\Model/Total257518Mesh_Lungs7113_Skel248326_Skin2079.obj'
    
    save_pass = r'E:\AAAA\AAAA\A_Science\Scoliosis\Model\PreTreat/TPS_Skin2079.obj'

    mesh_Skin = pv.read(Skin_file)
    mesh_Mode = pv.read(Mode_file)
    

    #SSM_vertsSuf, SSM_triSuf,Py_TRISuf = GetPtsFromSSM(SSM_file)
    #SSM_vertstal, SSM_trital, Py_TRITal = GetPtsFromSSM(SSM_file, type='All')
    #print('SSM_trital.shape = ', SSM_trital.shape)

    patient_file = r'E:\AAAA\AAAA\A_Science\Scoliosis\Patient\Zhangluyu/Cut_ZLY.stl'
    mesh_patient = pv.read(patient_file)
    mesh_patient = mesh_patient.decimate_boundary(target_reduction=1-(1500/mesh_patient.n_points))
    verts_patient = np.array(mesh_patient.points, dtype='float64')
    
    # 输入参数设置
    TPS_input1 = model_verts.copy()
    TPS_input2 = patient_file

    Model_TPS = SSM_TPS(TPS_input1, TPS_input2)


#----------------------------对合并后模型做X光投影---------------------------
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

    # X0_2Dpoint, Y0_2Dpoint, ConePrj_img = ConePrj_Mesh(model_verts, model_tri, Dtctr_in_world, 
    #                                                        xvec_in_world, yvec_in_world, Src_in_world)            
    








    time.sleep(3)


                # p = pv.Plotter()
                # p.background_color = 'w'                 
                
                # p.add_mesh(b, color="green", show_edges=True, opacity=1.0)
                # p.show()
    
