import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
import argparse
import skimage
import skimage.io as io
from skimage.color import rgb2hsv
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, concatenate, Lambda
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.optimizers import SGD, Adam
import matplotlib.patches as patches
import math
from keras.models import model_from_json
import open3d as o3d
import socket

filename = 'D:\\yangdata\\TrainData\\3rgb.png'		
filename_d = 'D:\\yangdata\\TrainData\\3d.png'
tfilename = 'D:\\yangdata\\pcd_grasp_img\\test_Color_fix1.png'		
tfilename_d = 'D:\\yangdata\\pcd_grasp_img\\test2_Deep_fix1.png'

data_file_X = open("D:\\yangdata\\pcd_grasp_img\\cloud_X.txt")
data_file_Y = open("D:\\yangdata\\pcd_grasp_img\\cloud_Y.txt")
data_file_Z = open("D:\\yangdata\\pcd_grasp_img\\cloud_Z.txt")	

def load_rgb(filename):
    img = image.load_img(filename, target_size = (224, 224))
    x = image.img_to_array(img) / 255
    return x
    
def load_depth(filename):
    im = image.load_img(filename, target_size=(224, 224))
    im = image.img_to_array(im)
    return im
    
def to_hsv(rgb, d):
    hsv = []
    hsv = rgb2hsv(rgb) * 255
    img = np.empty([1, 224, 224, 3], dtype = float)
    img[0, :, :, 0:2] = np.array(hsv)[:, :, 0:2]
    img[0, :, :, 2] = np.array(d)[:, :, 1]
    return img
    
def grasp_to_bbox(x, y, w, h, theta):
    theta = math.radians(theta)
    edge1 = (x -w/2*tf.cos(theta) +h/2*tf.sin(theta), y -w/2*tf.sin(theta) -h/2*tf.cos(theta))
    edge2 = (x +w/2*tf.cos(theta) +h/2*tf.sin(theta), y +w/2*tf.sin(theta) -h/2*tf.cos(theta))
    edge3 = (x +w/2*tf.cos(theta) -h/2*tf.sin(theta), y +w/2*tf.sin(theta) +h/2*tf.cos(theta))
    edge4 = (x -w/2*tf.cos(theta) -h/2*tf.sin(theta), y -w/2*tf.sin(theta) +h/2*tf.cos(theta))
    return [edge1, edge2, edge3, edge4]

def draw_bbox(img, s_img, ns_img, bbox):
    print(img.shape)
    #print(s_img.shape)
    print(ns_img.shape)
    ROW = ns_img.shape[0]
    COL = ns_img.shape[1]
	
    print(type(bbox[0][0]))
    p1 = (int(float(bbox[0][0])), int(float(bbox[0][1])))
    p2 = (int(float(bbox[1][0])), int(float(bbox[1][1])))
    p3 = (int(float(bbox[2][0])), int(float(bbox[2][1])))
    p4 = (int(float(bbox[3][0])), int(float(bbox[3][1])))	
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    temp_list=[]
    temp_list.append("A")
    for j in range(2):
        i_100 = int(p1[j]/100)
        i_10 = int((p1[j]-(i_100*100))/10)
        i_1 = int(p1[j]-i_100*100-i_10*10)
        temp_list.append(i_100)
        temp_list.append(i_10)	
        temp_list.append(i_1)
    for j in range(2):
        i_100 = int(p2[j]/100)
        i_10 = int((p2[j]-(i_100*100))/10)
        i_1 = int(p2[j]-i_100*100-i_10*10)
        temp_list.append(i_100)
        temp_list.append(i_10)	
        temp_list.append(i_1)
    for j in range(2):
        i_100 = int(p3[j]/100)
        i_10 = int((p3[j]-(i_100*100))/10)
        i_1 = int(p3[j]-i_100*100-i_10*10)
        temp_list.append(i_100)
        temp_list.append(i_10)	
        temp_list.append(i_1)	
    for j in range(2):
        i_100 = int(p4[j]/100)
        i_10 = int((p4[j]-(i_100*100))/10)
        i_1 = int(p4[j]-i_100*100-i_10*10)
        temp_list.append(i_100)
        temp_list.append(i_10)	
        temp_list.append(i_1)		
        temp_list.append("D")		
    print(temp_list) 		
    four_points = open("D:\\LocationAll.txt", "w")	
	
    for k in temp_list:        
        four_points.write(str(k))	

    four_points.close()
    #point_x_list = [int(float(bbox[0][0])),int(float(bbox[1][0])),int(float(bbox[2][0])),int(float(bbox[3][0]))]
    #point_y_list = [int(float(bbox[0][0])),int(float(bbox[0][1])),int(float(bbox[0][2])),int(float(bbox[0][3]))]
    center_x = int((p1[0]+p3[0]) / 2)
    center_y = int((p1[1]+p3[1]) / 2)
    print("left img grasp center : ",center_x,center_y)
    
    cv2.line(img, p1, p2, (0, 0, 255))
    cv2.line(img, p2, p3, (255, 0, 0))
    cv2.line(img, p3, p4, (0, 0, 255))
    cv2.line(img, p4, p1, (255, 0, 0))
    #cv2.circle(img,(center_x,center_y),1,(0, 0, 0),2)
	

    ration1 = ns_img.shape[1]/224
    ration2 = ns_img.shape[0]/224
    print("right img grasp center : ",int(center_x*ration1),int(center_y*ration2))
    pcd_num = ((int(center_x*ration1)-1)*ns_img.shape[1])+int(center_y*ration2)
    print(pcd_num)
    #print("Normal vector position: ",(int(center_x*ration1))*(int(center_x*ration2)))
    cv2.line(ns_img, (int(p1[0]*ration1),int(p1[1]*ration2)), (int(p2[0]*ration1),int(p2[1]*ration2)), (0, 0, 255))
    cv2.line(ns_img, (int(p2[0]*ration1),int(p2[1]*ration2)), (int(p3[0]*ration1),int(p3[1]*ration2)), (255, 0, 0))
    cv2.line(ns_img, (int(p3[0]*ration1),int(p3[1]*ration2)), (int(p4[0]*ration1),int(p4[1]*ration2)), (0, 0, 255))
    cv2.line(ns_img, (int(p4[0]*ration1),int(p4[1]*ration2)), (int(p1[0]*ration1),int(p1[1]*ration2)), (255, 0, 0))	
    #cv2.circle(ns_img,(int(cen#ter_x*ration1),int(center_y*ration2)),1,(0, 0, 0),2)
    #cv2.circle(ns_img,(int(center_x*(ns_img.shape[1]/224)),int(center_y*(ns_img.shape[0]/224))),1,(0, 0, 0),2)
    return pcd_num

def get_pts(n2):
    data_X = []
    data_Y = []
    data_Z = []
    for linex in data_file_X:	
        data_X.append(linex)
    data_file_X.close()

    for liney in data_file_Y:	
        data_Y.append(liney)
    data_file_Y.close()

    for linez in data_file_Z:	
        data_Z.append(linez)
    data_file_Z.close()   
	
    data_X = list(map(float,data_X))
    data_Y = list(map(float,data_Y))
    data_Z = list(map(float,data_Z))
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    data_Z = np.array(data_Z)
    #print(data_X.shape)
    data_XYZ = np.vstack((data_X,data_Y,data_Z))
    data_XYZ = data_XYZ.tolist()
    data_XYZ = list(map(list,zip(*data_XYZ)))
    data_XYZ = np.array(data_XYZ)
    data_XYZ_p = np.around(data_XYZ, decimals=6, out=None)
    print("pcd_arrary_X: ",data_XYZ_p[n2][0])
    print("pcd_arrary_Y: ",data_XYZ_p[n2][1])
    print("pcd_arrary_Z: ",data_XYZ_p[n2][2])	
	
    pcd_X = open("D:\\Center_Location_X.txt", "w")	
    pcd_X.write("A")
    pcd_X.write(str(data_XYZ_p[n2][0]))	
    pcd_X.write("D")	
    pcd_X.close()
	
    pcd_Y = open("D:\\Center_Location_Y.txt", "w")	
    pcd_Y.write("A")
    pcd_Y.write(str(data_XYZ_p[n2][1]))	
    pcd_Y.write("D")	
    pcd_Y.close()
	
    pcd_Z = open("D:\\Center_Location_Z.txt", "w")	
    pcd_Z.write("A")  
    pcd_Z.write(str(data_XYZ_p[n2][2]))	
    pcd_Z.write("D")	
    pcd_Z.close()	
    return data_XYZ	
    
def plot_ply(n1):
    N1 = n1
    d = get_pts(N1)
    #print(d)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d)	
    o3d.io.write_point_cloud("3D_ponitCloud.ply", pcd)	
    pcd_load = o3d.io.read_point_cloud("3D_ponitCloud.ply")
    #o3d.visualization.draw_geometries([pcd_load])   

    pcd_load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=10))
    #print("normal_vectorALL: ",pcd_load.normals[n1])
    vectorarrary = np.around(pcd_load.normals, decimals=6, out=None)
    print("normal_vector_X: ",vectorarrary[n1][0])
    print("normal_vector_Y: ",vectorarrary[n1][1])
    print("normal_vector_Z: ",vectorarrary[n1][2])

    vector_X = open("D:\\Normalvector_X_F.txt", "w")	
    vector_X.write("A")
    vector_X.write(str(vectorarrary[n1][0]))	
    vector_X.write("D")	
    vector_X.close()
	
    vector_Y = open("D:\\Normalvector_Y_F.txt", "w")	
    vector_Y.write("A")
    vector_Y.write(str(vectorarrary[n1][1]))	
    vector_Y.write("D")	
    vector_Y.close()
	
    vector_Z = open("D:\\Normalvector_Z_F.txt", "w")	
    vector_Z.write("A")  
    vector_Z.write(str(vectorarrary[n1][2]))	
    vector_Z.write("D")	
    vector_Z.close()    


if __name__ == '__main__':

    #Load Model
    print('loading model...')
    
    json_file = open("model_fT.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model_fT.h5")

    rgb_image = load_rgb(filename)
    d_image = load_depth(filename_d)
    hsv_image = to_hsv(rgb_image, d_image)
    print('in process...')
    test = model.predict(hsv_image)
    print(test)
    bbox_model = grasp_to_bbox(test[0][0], test[0][1], test[0][2], test[0][3], test[0][4])
    square_img_path = filename
    N_square_img_path = filename
    square_img = cv2.imread(square_img_path)
    N_square_img = cv2.imread(N_square_img_path)
    PCD = draw_bbox(rgb_image, square_img ,N_square_img,bbox_model)
    img_result1 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    img_result2 = cv2.cvtColor(N_square_img, cv2.COLOR_RGB2BGR)      

    plot_ply(PCD)
    plt.subplot(121)
    plt.imshow(img_result1)
    plt.subplot(122)
    plt.imshow(img_result2)   
    plt.show()