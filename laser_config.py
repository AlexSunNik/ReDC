import math
import time
import random
import json
import pprint
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import xml.etree.ElementTree as ET

from pathlib import Path



config_path = '/shared/rsaas/common/Kitti/velodynecalib_S2_factory_flatness_intensity.xml'
def laser_calib_params(dataset="kitti", path=config_path):

    if dataset == "kitti":

        tree = ET.parse(path)
        root = tree.getroot()
        laser_params = np.zeros((64, 9))

        for laser in root.iter('px'):
            id_ = int(laser.find('id_').text)
            rotCorrection_ = float(laser.find('rotCorrection_').text)
            vertCorrection_ = float(laser.find('vertCorrection_').text)
            distCorrection_ = float(laser.find('distCorrection_').text) * 1e-2
            distCorrectionX_ = float(laser.find(
                'distCorrectionX_').text) * 1e-2
            distCorrectionY_ = float(laser.find(
                'distCorrectionY_').text) * 1e-2
            vertOffsetCorrection_ = float(laser.find(
                'vertOffsetCorrection_').text) * 1e-2
            horizOffsetCorrection_ = float(laser.find(
                'horizOffsetCorrection_').text) * 1e-2
            focalDistance_ = float(laser.find('focalDistance_').text) * 1e-2
            focalSlope_ = float(laser.find('focalSlope_').text)

            laser_params[id_] = [rotCorrection_, vertCorrection_, distCorrection_, distCorrectionX_, 
                                distCorrectionY_, vertOffsetCorrection_, horizOffsetCorrection_, 
                                focalDistance_, focalSlope_]

    else:
        laser_params = np.zeros((16, 9))
        vertCorrection = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
        vertOffsetCorrection = [11.2, -0.7, 9.7, -2.2, 8.1, -3.7, 6.6, -5.1, 5.1, -6.6, 3.7, -8.1, 2.2, -9.7, 0.7, -11.2]
        vertOffsetCorrection = np.array(vertOffsetCorrection) * 1e-3
        laser_params[:, 1] = vertCorrection
        laser_params[:, 5] = vertOffsetCorrection

    return laser_params




