# -*- coding: utf-8 -*-
"""
Time: 2026/4/24 00:31
Author: lcqin111
Version: V1.0
File: demo.py
Description: demo for testing the toolbox
"""
from fabopsy_ufanet.core import Detector
from PIL import Image
import numpy as np

model = Detector('./20260422_195809_50.safetensors')
img_path = './test2.png'

img = np.array(Image.open(img_path).convert("RGB"))
cls_pred, au_pred, valence_pred, arousal_pred = model.detect(img)
print('cls:', cls_pred)
print('au:', au_pred)
print('valence:', valence_pred)
print('arousal:', arousal_pred)