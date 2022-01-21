import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import copy


data_file_X = open("D:\\yangdata\\pcd_grasp_img\\cloud_X.txt")
data_file_Y = open("D:\\yangdata\\pcd_grasp_img\\cloud_Y.txt")
data_file_Z = open("D:\\yangdata\\pcd_grasp_img\\cloud_Z.txt")

#data_file_X = open("D:\\test_X.txt")
#data_file_Y = open("D:\\test_Y.txt")
#data_file_Z = open("D:\\test_Z.txt")

def get_pts():
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
    #print(data_XYZ.shape)
	
    return data_XYZ



def plot_ply():
    d = get_pts()
    print(d)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d)	
    o3d.io.write_point_cloud("3D_ponitCloud.ply", pcd)	
    pcd_load = o3d.io.read_point_cloud("3D_ponitCloud.ply")
    #o3d.visualization.draw_geometries([pcd_load])   

    pcd_load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    #print(pcd_load.normals[7421])
    
    #final = o3d.utility.Vector3dVector(pcd_load.normals[7420])
    #o3d.visualization.draw_geometries([pcd_load.normals[7420]])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print("Paint the 1500th point red.")
    pcd_load.colors[4196] = [1, 0, 0]

    print("Find its 200 nearest neighbors, paint blue.")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd_load.points[4196], 30)
    np.asarray(pcd_load.colors)[idx[1:], :] = [0, 0, 1]

    print("Find its neighbors with distance less than 0.2, paint green.")
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd_load.points[4196], 10)
    np.asarray(pcd_load.colors)[idx[1:], :] = [0, 1, 0]

    print("Visualize the point cloud.")
    o3d.visualization.draw_geometries([pcd_load])
	   
	
if __name__ == '__main__':
	plot_ply()


