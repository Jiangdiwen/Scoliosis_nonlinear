import numpy as np
import pyvista as pv
from numpy import linalg as la
import time
from sklearn.neighbors import NearestNeighbors


#import sys
#import numpy
#transform default truncated representation to full representation when check all the numbers in an array and so on.
#numpy.set_printoptions(threshold=sys.maxsize) #


def Calc_SSMTsfm_2PtSets(Vy, SSM, deform_TH=0, PartialList=0, numPC=0, ind_PCs=0, Sz_Target=0,
                         noSimTsfm=0, noScaling=0, coef_range=0, SmpRateTPS=2,
                         VertFittingWeights=0, isDisp=0, DimFit=0):

    num_Verts = int(np.size(Vy) / 3)
    num_Verts1 = int(np.size(SSM['Verts_mean_Skin']) / 3)
    num_Verts2 = int(np.size(SSM['Verts_mean_total']) / 3)
    numPC = numPC if numPC != 0 else SSM['numPC']
    ind_PCs = ind_PCs if ind_PCs != 0 else np.arange(numPC)

    # if only fit one or two dimensions
    if DimFit != 0:
        vv = np.zeros((num_Verts1, 3))
        for idim in range(3):
            if DimFit[idim] != 0:
                vv[:, idim] = 1
        indim_fit = np.where(vv.flatten() != 0)

    if PartialList == 0:
        if noSimTsfm == 0:
            Vx_mean = np.reshape(SSM['Verts_mean_Skin'], (num_Verts, 3))  # mean shape of the SSM
            ox = np.sum(Vx_mean, axis=0) / num_Verts1 # the centroid of the mean shape
            Vx_centered = Vx_mean - ox  # decentrerization to the SSM
            sx = np.sqrt(np.sum(Vx_centered**2)) # scale of the SSM

            oy = np.mean(Vy, axis=0)  # the centroid of the target points
            Vy_norm = Vy - oy # decenterization of the target points
            sy = np.sqrt(np.sum(Vy_norm**2))   # scale of the target point
            scl_x2y = (Sz_Target / sx) if Sz_Target != 0 else sy/sx    #  the scaling factor from the target points to the SSM
            Vy_norm = Vy_norm / scl_x2y    # scale the target points

            # rotate the target points
            W, U, V = la.svd(np.dot(Vy_norm.T, Vx_centered))
            ROT_y2x = np.dot(W, V)
            Rot_x2y = ROT_y2x.T
            scaling = scl_x2y if noScaling != 0 else 1.0
            Vy_norm = np.dot(Vy_norm, ROT_y2x) * scaling + ox
        else:
            ox = -1
            oy = -1
            Rot_x2y = -1
            sx = -1
            sy = -1
            Vy_norm = Vy

        # deform the model, i.e. solve a linear system
        a = SSM['Eigenvect_Skin'][ind_PCs, :].T
        Vy_norm = Vy_norm.flatten()
        b = Vy_norm - SSM['Verts_mean_Skin']

        # add the vertex weights
        if VertFittingWeights != 0:
            n_usedPC = np.size(ind_PCs)
            VerFitW = np.zeros((num_Verts1, 3))
            VerFitW = VerFitW + VertFittingWeights.T
            wts = VerFitW.flatten()
            atmp = np.zeros((3*num_Verts1, n_usedPC))
            a = a * (wts + atmp)
            b = b * wts
        if DimFit != 0:
            a = a[indim_fit, :]
            b = b[indim_fit]

        coef, _, _, _ = la.lstsq(a, b)
    else:
        nv_Partial = np.size(PartialList)
        ind_fit = np.zeros((nv_Partial, 3)).astype(int)
        for idim in range(3):
            ind_fit[:, idim] = PartialList * 3 + idim
        if noSimTsfm == 0:
            Vx_mean = np.reshape(SSM['Verts_mean_Skin'][ind_fit],(nv_Partial, 3))
            ox = np.sum(Vx_mean, axis=0)/nv_Partial
            Vx_centered = Vx_mean - np.tile(ox, (nv_Partial, 1))
            sx = np.sqrt(np.sum(Vx_centered**2))

            oy = np.sum(Vy, axis=0)/nv_Partial
            Vy_norm = Vy - np.tile(oy, (nv_Partial, 1))
            sy = np.sqrt(np.sum(Vy_norm**2))
            scl_x2y = sy / sx
            Vy_norm = Vy_norm / scl_x2y
            W, U, V = la.svd(np.dot(Vy_norm.T, Vx_centered))
            ROT_y2x = np.dot(W, V)
            Rot_x2y = ROT_y2x.T
            scaling = scl_x2y if noScaling != 0 else 1.0
            Vy_norm = np.dot(Vy_norm, ROT_y2x) * scaling + np.tile(ox, (nv_Partial, 1))
        else:
            ox = -1
            oy = -1
            Rot_x2y = -1
            sx = -1
            sy = -1
            Vy_norm = Vy

        # deform model
        a = SSM['Eigenvect_Skin'][ind_PCs, :]
        a = a[:, ind_fit].T
        Vy_norm = Vy_norm.flatten()
        b = Vy_norm - SSM['Verts_mean_Skin'][ind_fit]

        if DimFit != 0:
            a = a[indim_fit, :]
            b = b[indim_fit]

        coef, _, _, _ = la.lstsq(a, b)

    if deform_TH != 0:
        d_max = np.sqrt(-2 * np.log(deform_TH))
        coeftmp = np.dot(coef, la.inv(SSM['cov']))
        d_deform = np.sqrt(np.dot(coeftmp, coef.T)) 
        if d_deform > d_max: 
            coef = coef * (d_max / d_deform[0])
    #me 小改动，因为缺少特征值
    coef_range = 0
    if coef_range != 0:
        maxinum = np.maximum(coef, -coef_range * SSM['Eigenval'][ind_PCs]) # 取两个同样尺寸数组对应位置元素的较大值组成一个数组
        coef = np.minimum(maxinum, coef_range * SSM['Eigenval'][ind_PCs])  # 取两个同样尺寸数组对应位置元素的较小值组成一个数组

    if noSimTsfm == 0:
        Mc = np.eye(4, 4)
        for id in range(3):
            Mc[3, id] = -ox[id]
        Mr = np.eye(4, 4)
        Mr[0:3, 0:3] = Rot_x2y
        Ms = np.eye(4, 4)
        for id in range(3):
            Ms[id, id] = scl_x2y
        Md = np.eye(4, 4)
        for id in range(3):
            Md[3, id] = oy[id]
        tmp1 = np.dot(Mc, Mr)
        tmp2 = np.dot(tmp1, Ms)
        Sim = np.dot(tmp2, Md)
    else:
        Sim = np.eye(4, 4)
    # 约束系数不能超出[-3,3]
    # num = 0
    # for i in coef:
    #     if i > 3:
    #         coef[num] = 3
    #     elif i < -3:
    #         coef[num] = -3
    #     num += 1

    # the fitted points
    Vx0 = SSM['Verts_mean_total'] + np.dot(coef, SSM['Eigenvect_total'][ind_PCs, :])
    Vx0 = np.reshape(Vx0, (num_Verts2, 3))
    if noSimTsfm == 0:
        Vx = np.hstack((Vx0, np.zeros((num_Verts2, 1)) + 1))
        Vx = np.dot(Vx, Sim)
        Vx = Vx[:, 0:3]
    else:
        Vx = Vx0

    if SmpRateTPS != 0:
        pass
    if isDisp != 0:
        pass

    Registration_Model = {"Vx":Vx, "Vx0":Vx0, "coef":coef, "Sim":Sim, "ox":ox, "oy":oy, "Rot_x2y":Rot_x2y,
                          "sx":sx, "sy":sy, "numPC":numPC }
    return Registration_Model



def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    #assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1, radius=0.1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(SSM, Target_Verts, indSmp, Target_Faces, init_pose=None, max_iterations=20, tolerance=0.00001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''
    SSM_ICP = SSM.copy()
    # get number of dimensions
    m = SSM['Verts_mean'].reshape(-1, 3).shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.copy(SSM['Verts_mean'].reshape(-1, m))
    centroid_src = np.mean(src, axis=0)
    Stadsrc = src - centroid_src
    Scaled_src = np.sqrt(np.sum(Stadsrc**2))

    B = np.copy(Target_Verts[indSmp, :])
    centroid_T = np.mean(Target_Verts[indSmp, :], axis=0)
    StadT = Target_Verts[indSmp, :] - centroid_T
    Scaled_T = np.sqrt(np.sum(StadT**2))
    dstt = (StadT / Scaled_T)*Scaled_src
    dst = dstt  #[indSmp, :]

    # 测试SSM和待配准目标初始位置是否基本对齐 ############################################
    surfsrc = pv.PolyData(src, SSM['Conn_mean'])
    surfdst = pv.PolyData(dst)
    plt = pv.Plotter()
    plt.add_mesh(surfsrc, color="blue", opacity=0.75, show_edges=True, point_size=3, render_points_as_spheres=True, label=r"SSM")
    plt.add_mesh(surfdst, color="red", opacity=0.75, show_edges=True, point_size=3, render_points_as_spheres=True, label=r"Target object")
    #plt.add_point_labels([surfsrc.center,], ['Center',], point_color='yellow', point_size=10)
    #plt.add_point_labels([surfdst.center,], ['Center',], point_color='orange', point_size=10)

    plt.show()
    #####################################################################################
    #src = src - surfsrc.center
    #dst = dst - surfdst.center



    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0


    #Store_index = np.zeros((max_iterations, B.shape[0]))
    for i in range(max_iterations):

        # find the nearest neighbors between the current source and destination points
        start = time.time()
        distances, indices = nearest_neighbor(src, dst)
        #Store_index[i] = indices
        #print('indices: ', indices)
        total_time = time.time() - start
        print('The time to find nearest neighbors in one iteration is: {:.3}'.format(total_time))
        dst = B

        # compute the transformation between the current source and nearest destination points
        start = time.time()
        # T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        Registration_Model = Calc_SSMTsfm_2PtSets(dst[indices, :m], SSM_ICP, coef_range=0.1)
        total_time = time.time() - start
        print('One transformation time is: {:.3}'.format(total_time))

        # update the current source
        # T = Registration_Model['Sim']

        # 将变换后的模型统一化到平均模型尺寸
        centroid_Vx = np.mean(Registration_Model['Vx'], axis=0)
        StadVx = Registration_Model['Vx'] - centroid_Vx
        Scaled_Vx = np.sqrt(np.sum(StadVx**2))
        src = (StadVx / Scaled_Vx)*Scaled_src

        # 将待配准模型统一化到平均模型尺寸
        centroid_dst = np.mean(dst, axis=0)
        Staddst = dst - centroid_dst
        Scaled_dst = np.sqrt(np.sum(Staddst**2))
        dst = (Staddst / Scaled_dst)*Scaled_src


        SSM_ICP['Verts_mean'] = src.flatten()
        src1 = Registration_Model['Vx']


        surf = pv.PolyData(src1, SSM['Conn_mean'])
        surf['vectors'] = dst[indices, :m] - src1
        arrows = surf.glyph(orient='vectors', scale=False, factor=0.2)
        surf2 = pv.PolyData(Target_Verts, Target_Faces)
        #normal = Normals[:, :, 11].glyph(orient="ABC field vectors", factor=50.0)


        plt = pv.Plotter()

        plt.add_mesh(surf, color="blue", style='surface', show_edges=True, line_width=0.5, opacity=0.5, label=r"Target object")
        plt.add_mesh(pv.PolyData(surf.points), color='orange', point_size=3, render_points_as_spheres=True)
        plt.add_mesh(arrows, color='white', line_width=1, lighting=False)
        plt.add_mesh(surf2, color="red", style='surface', line_width=0.5, opacity=0.5, label=r"SSM")
        plt.add_mesh(pv.PolyData(surf2.points), color='red', point_size=3, render_points_as_spheres=True)
        #plt.show_grid()
        #plt.show()


        # check error
        #distances = np.sqrt(np.sum((Registration_Model['Vx'] - dst)**2, axis=1))
        mean_error = np.mean(distances)
        print('The average distance change between the last registration point cloud and the current registration point cloud is {}'.format(prev_error - mean_error))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    # Registration_Model = Calc_SSMTsfm_2PtSets(src[:m,:].T, SSM_ICP)
    # T = Registration_Model['Sim']
    # T,_,_ = best_fit_transform(A, src[:m,:].T)

    return Registration_Model
