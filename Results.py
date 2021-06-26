#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import cv2 as cv
import numpy as np
root_dir="C:/Users/Dell/Desktop/topcoder/test/"#change this to your current directory containing the validation images
pdf=pd.read_csv(root_dir+"validationdataset.csv")
y_finalu=[]
y_file=[]
shape=(140,140)
shape1=(140,140)
json_file = open(root_dir+'Densenet101.json', 'r')#Loads deep learning model Densenet201
json_file1=open(root_dir+'Densenet169_17th_2.json','r')#Loads deep Learning model Densenet169
#json_file2=open(root_dir+'Densenet121_17th.json','r')#Loads deep Learning model Densenet121
loaded_model_json = json_file.read()
loaded_model_json1=json_file1.read()
#loaded_model_json2=json_file2.read()
json_file.close()
from keras.models import model_from_json
loaded_model =model_from_json(loaded_model_json)
loaded_model1=model_from_json(loaded_model_json1)
#loaded_model2=model_from_json(loaded_model_json2)
# load weights into new model
loaded_model.load_weights(root_dir+"Densenet101.h5")#Loads weights of models Densenet201
loaded_model1.load_weights(root_dir+"Densenet169_17th_2.h5")
#loaded_model2.load_weights(root_dir+"Densenet121_17th.h5")
print("Loaded models from disk")


# In[ ]:



import os
y_final12=[]
shape=(140,140)
img1=input("Please input the path of the directory containing all the images")
for t in os.listdir(img1):
    cvb=cv.resize(cv.imread(img1+"/"+t),shape)/255
    img = np.expand_dims(cvb, axis=0)
    g=np.argmax(np.squeeze((loaded_model.predict(img)))+np.squeeze((loaded_model1.predict(img))))
    if(g==0):
        y_final12.append("noshoes")
    elif(g==1):
        y_final12.append("boots")
    elif(g==2):
        y_final12.append("flipflops")
    elif(g==3):
        y_final12.append("loafers")
    elif(g==4):
        y_final12.append("Sandals")
    elif(g==5):
        y_final12.append("Soccershoes")
    elif(g==6):
        y_final12.append("Sneakers")
print(y_final12)


