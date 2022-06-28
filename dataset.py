import os
import torch
import numpy as np
from torch.utils.data import Dataset

class_to_id = {'DontCare':0,'Car':1}#,'Van':2,'Truck':3,'Pedestrian':4,'Person_sitting':5,'Cyclist':6,'Tram':7,'Misc':8}

class kitti_dataset(Dataset):
	def __init__(self,dataset_dir):
		super().__init__()
		self.dataset_dir = dataset_dir
		self.pc_dir = self.dataset_dir + "/velodyne/"
		self.calib_dir = self.dataset_dir + "/calib/"
		self.label_dir = self.dataset_dir + "/label_2/"
		self.pointcloud_list = os.listdir(dataset_dir+'/velodyne/')

	def __len__(self):
		return len(self.pointcloud_list)

	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		pointcloud = torch.tensor(np.fromfile(self.dataset_dir+'/velodyne/'+self.pointcloud_list[idx],dtype=np.float32).reshape(-1,4))
		target = self.get_targets(self.pointcloud_list[idx])
		return pointcloud,target,self.pointcloud_list[idx]

	def get_targets(self,file_name):

		calib_dict = self.load_kitti_calib(self.calib_dir+file_name[:-3]+'txt')
		targets_file = open(self.label_dir+file_name[:-3]+'txt','r')
		target_strs = targets_file.readlines()
		targets = []
		for line in target_strs:
			gt_bbox = []
			annotation = line.split('\n')[0]
			annotation = annotation.split(' ')
			
			if annotation[0] != 'Car':
				continue
			cls_target = [0]*len(class_to_id.keys())
			cls_target[class_to_id[annotation[0]]] = 1 

			gt_bbox.append(float(annotation[8])) # h
			gt_bbox.append(float(annotation[9])) # w
			gt_bbox.append(float(annotation[10])) # l
			gt_bbox.append(float(annotation[11])) # x
			gt_bbox.append(float(annotation[12])) # y
			gt_bbox.append(float(annotation[13])) # z
			gt_bbox.append(float(annotation[14])) # theta

			gt_bbox[3:6] = self.project_cam2velo(calib_dict['Tr_velo2cam'],gt_bbox[3:6])

			# Uncomment for bouding box rotated form
			# gt_bbox = bbox_format(gt_bbox)

			# targets.append(cls_target + gt_bbox[:-1])
			targets.append(np.array(cls_target + gt_bbox,dtype=np.float32))

		targets = torch.tensor(np.array(targets,dtype=np.float32),dtype=torch.float32)
		if targets.size()[0]==0:
			targets = torch.zeros(1,9)
		return targets

	def load_kitti_calib(self,calib_file):
		"""
		load projection matrix
		"""
		with open(calib_file) as fi:
			lines = fi.readlines()
			assert (len(lines) == 8)

		obj = lines[0].strip().split(' ')[1:]
		P0 = np.array(obj, dtype=np.float32)
		obj = lines[1].strip().split(' ')[1:]
		P1 = np.array(obj, dtype=np.float32)
		obj = lines[2].strip().split(' ')[1:]
		P2 = np.array(obj, dtype=np.float32)
		obj = lines[3].strip().split(' ')[1:]
		P3 = np.array(obj, dtype=np.float32)
		obj = lines[4].strip().split(' ')[1:]
		R0 = np.array(obj, dtype=np.float32)
		obj = lines[5].strip().split(' ')[1:]
		Tr_velo_to_cam = np.array(obj, dtype=np.float32)
		obj = lines[6].strip().split(' ')[1:]
		Tr_imu_to_velo = np.array(obj, dtype=np.float32)

		return {'P2': P2.reshape(3, 4),
			'R0': R0.reshape(3, 3),
			'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

	def project_cam2velo(self,projection_mat,center):
		'''
			The ground truth bounding box in the label file is wrt to
			the camera coordinate system which has to be changed to the lidar's
		'''
		x,y,z = center
		cam = np.ones([4,1])
		cam[0] = x
		cam[1] = y
		cam[2] = z

		T = np.zeros([4,4],dtype=np.float32)
		T[:3, :] = projection_mat
		T[3, 3] = 1
		T_inv = np.linalg.inv(T)
		lidar_loc_ = np.dot(T_inv, cam)
		lidar_loc = lidar_loc_[:3]
		return lidar_loc

def bbox_format(bbox):
	'''
		The bbox is rotated by the orientation angle
	'''
	ry = bbox[-1]
	angle = -ry -np.pi/2

	if angle >= np.pi:
		angle -= np.pi
	if angle < -np.pi:
		angle = 2*np.pi + angle

	rotMat = np.array([
		[np.cos(angle), -np.sin(angle), 0.0],
		[np.sin(angle), np.cos(angle), 0.0],
		[0.0, 0.0, 1.0]])

	bbox_dims = np.array([[bbox[2]],[bbox[1]],[bbox[0]]])
	bbox_dims = np.dot(rotMat, bbox_dims)
	bbox[2] = bbox_dims[0]
	bbox[1] = bbox_dims[1]
	bbox[0] = bbox_dims[2]
	
	return bbox
