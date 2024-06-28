'''
Author: Yanxin_Jiang
Date: 2022-04-16 15:35:01
LastEditors: Yanxin_Jiang
LastEditTime: 2022-06-06 23:07:34
Description: 

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script Name : elastix_f.py
Author : Tian Yang
Created : 2020/10/06
Last modified :
Version :
Modification :
Description :
    PURPOSE :
    INPUTS :
    OUTPUTS :
'''

import numpy as np
import re
import math
import os
import sys
import subprocess
import shutil
import fileinput
import SimpleITK as sitk


def elastix_f(fixedVolume, movingVolume, outputDir):
    PATH = r'E:/Useful_software/elastix-5.0.1-win64'
    parameterDir = os.path.join(PATH, "parameters")
    parameterFile0 = os.path.join(parameterDir, "parameters_Affine.txt")
    # parameterFile0 = os.path.join(parameterDir, "Elastix_3DSimilarityMI.txt")
    # parameterFile0 = os.path.join(parameterDir, "Elastix_3DRigidMI-noOutputImg.txt")
    parameterFile1 = os.path.join(parameterDir, "parameters_BSpline.txt")

    ELASTIX = os.path.join(PATH, 'elastix.exe')
    #elastixCommand = "%s -f %s -m %s -out %s -p %s -p %s" % (
       # ELASTIX, fixedVolume, movingVolume, outputDir, parameterFile0,
       # parameterFile1)
   #./elastix.exe -f 'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/pretreatment/wuyifan.nii' -m 'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/pretreatment/XConep.mhd' -out 'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/2Dregistration_result/' -p 'E:/Useful_software/elastix-5.0.1-win64/parameters/parameters_Rigid.txt'
    elastixCommand = "%s -f %s -m %s -out %s -p %s -p %s" % (
        ELASTIX, fixedVolume, movingVolume, outputDir, parameterFile0, parameterFile1)
    p = subprocess.Popen(elastixCommand.split(" "),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    p.wait()
    out = out.splitlines()
    # print(out)
    success = False
    for line in out:
        if ("Total time elapsed" in line.decode()):
            seconds = line.decode().replace("Total time elapsed: ", "")
            success = True
    print(seconds)


def transformix_f(point, outputDir, parameterTransformix):
    PATH = r'E:/Useful_software/elastix-5.0.1-win64'    
    Transformix = os.path.join(PATH, 'transformix.exe')
    # parameterTransformix = os.path.join(right, "TransformParameters.0.txt")
    transformixCommand = "%s -def %s -out %s -tp %s" % (
        Transformix, point,outputDir, parameterTransformix)
    p = subprocess.Popen(transformixCommand, stdout=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        print
        line,
    p.stdout.close()
    p.wait()
    print('Registration Complete!')
""""""

if __name__ == "__main__":
    f_mov = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\pretreatment\patients/BYM_Downsample.nii'
    f_fix = r'E:/AAAA/AAAA/SeniorYear/digital_human_Xray/ConePrj_Mesh_Data/XConepSSM_testSetwindow.nii'

    # mv = r"D:/SPM_test/7163/P00007163_FANGLI_StudyID0_SeriesNo9_SUV.nii"

    out_path = r'E:\AAAA\AAAA\SeniorYear\digital_human_Xray\2Dregistration_result\test_result/'

    elastix_f(f_fix, f_mov, out_path)
