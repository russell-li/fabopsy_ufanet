# -*- coding: utf-8 -*-
"""
Time: 2026/4/24 00:31
Author: lcqin111
Version: V1.0
File: demo.py
Description: demo for testing the toolbox
"""
from fabopsy_ufanet.core import Detector

model = Detector('./20260422_195809_50.safetensors')
cls_pred, au_pred, valence_pred, arousal_pred = model.detect('./test2.png')
print('cls:', cls_pred)
print('au:', au_pred)
print('valence:', valence_pred)
print('arousal:', arousal_pred)