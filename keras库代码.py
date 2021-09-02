# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:07:15 2021

@author: Administrator
"""



# -*- coding: UTF-8 -*-
 
import os, sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import keras
import pickle
import PIL.Image
import random
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
# 准备用来训练网络和测试的数据集
from dataset import DataSet

def scheduler(epoch):
    if epoch < 20:
        return 0.1
    if epoch < 40:
        return 0.01
    return 0.001


# 主程序
def main():
    # 输入图片为256x256，2个分类
    shape, classes = (64, 64, 3), 2
    log_filepath = './PSEres5-1922/' 
    # 调用keras的ResNet50模型
    model = keras.applications.resnet50.ResNet50(input_shape = shape, weights=None, classes=classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # 给出训练和测试数据
    X_train, Y_train, X_test, Y_test = DataSet()
    print('X_train shape : ', X_train.shape)
    print('Y_train shape : ', Y_train.shape)
    print('X_test shape : ', X_test.shape)
    print('Y_test shape : ', Y_test.shape)
    
    # 训练模型
    #num_classes        = 2
    batch_size         = 64
    epochs             = 50
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
#自适应学习率
    change_lr = LearningRateScheduler(scheduler)

    cbks = [change_lr,tb_cb]
    model.fit(X_train, Y_train,batch_size=batch_size,
                     #steps_per_epoch=iterations,
                     epochs=epochs,
                     #callbacks=[cbks,checkpoint],
                     callbacks=cbks,
                     validation_data=(X_test, Y_test)
                     )
    model.save(log_filepath + 'PSEresdense.h5')
 
if __name__ == "__main__":
    main()
