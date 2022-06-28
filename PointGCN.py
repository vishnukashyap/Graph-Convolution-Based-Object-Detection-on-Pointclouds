import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule
import math

class PointGCN(nn.Module):
	def __init__(self,use_xyz,num_classes):
		super(PointGCN,self).__init__()
		self.use_xyz = use_xyz
		self.num_classes = num_classes
		self._build_model()

	def _build_model(self):
		
		self.SA_module1 = PointnetSAModule(
			npoint=25000,
			radius=1.0,
			nsample=64,
			mlp=[1,64],
			use_xyz=self.use_xyz,
		)

		self.SA_module2 = PointnetSAModule(
			npoint=12500,
			radius=1.0,
			nsample=64,
			mlp=[64,128],
			use_xyz=self.use_xyz,
		)

		self.SA_module3 = PointnetSAModule(
			npoint=10000,
			radius=1.0,
			nsample=64,
			mlp=[128,256], use_xyz=self.use_xyz
		)

		# Reg Block
		self.Conv2d_1 = torch.nn.Sequential(*[torch.nn.Conv2d(256,128,(3,1),padding=(1,0),bias=True),torch.nn.BatchNorm2d(128),torch.nn.ReLU()])
		self.Conv2d_2 = torch.nn.Sequential(*[torch.nn.Conv2d(128,64,(3,1),padding=(1,0),bias=True),torch.nn.BatchNorm2d(64),torch.nn.ReLU()])
		self.regression_block = torch.nn.Conv1d(64,7,1)

		# Classification Block
		self.Conv1d_1 = torch.nn.Sequential(*[torch.nn.Conv1d(256,128,1),torch.nn.BatchNorm1d(128),torch.nn.ReLU()])
		self.Conv1d_2 = torch.nn.Sequential(*[torch.nn.Conv1d(128,64,1),torch.nn.BatchNorm1d(64),torch.nn.ReLU()])
		self.classification_block = torch.nn.Linear(64,self.num_classes)

	def _break_up_pc(self,pc):
		xyz = pc[...,0:3].contiguous()
		features = pc[...,3:].transpose(1,2).contiguous() if pc.size(-1) > 3 else None

		return xyz,features

	def forward(self,pointcloud):

		xyz,features = self._break_up_pc(pointcloud)
		
		xyz,features = self.SA_module1(xyz,features)

		xyz,features = self.SA_module2(xyz,features)

		xyz,features = self.SA_module3(xyz,features)

		# Classification Block where we take the features from the backbone and rund Conv1d to get the local features for better classifications
		cls_output = self.Conv1d_1(features)
		cls_output = self.Conv1d_2(cls_output)
		classification_output = self.classification_block(cls_output.permute(0,2,1))
		
		# Regression Block where we take the features output from the backbone and run a conv2d to get the global features for better predicitons
		reg_output = self.Conv2d_1(features.unsqueeze(-1)).squeeze(-1)
		reg_output = self.Conv2d_2(reg_output.unsqueeze(-1)).squeeze(-1)
		regression_output = self.regression_block(reg_output).permute(0,2,1)

		del reg_output
		del cls_output
		del features
		torch.cuda.empty_cache()

		return xyz,classification_output,regression_output

	def encode_bbox(self,xyz_tensor,bbox_tensor):
		'''
			Bounding box encoding is done as described in the paper PointGNN
			The scaling factors for the time being is initialized to 1
			Other values to be verified once initial training is done
		'''
		batch_bbox = []
		for b in range(bbox_tensor.size()[0]):
			encoded_bboxs = []
			for i in range(bbox_tensor.size()[1]):
				encoded_bbox = []
				xyz = xyz_tensor[b,i,:]
				bbox = bbox_tensor[b,i,:]
				encoded_bbox.append(bbox[3]-xyz[0])
				encoded_bbox.append(bbox[4]-xyz[1])
				encoded_bbox.append(bbox[5]-xyz[2])
				if bbox[0]>0.:
					encoded_bbox.append(math.log(bbox[0]))
				else:
					encoded_bbox.append(0.)
				if bbox[1]>0.:
					encoded_bbox.append(math.log(bbox[1]))
				else:
					encoded_bbox.append(0.)
				if bbox[2]>0.:
					encoded_bbox.append(math.log(bbox[2]))
				else:
					encoded_bbox.append(0.)
				encoded_bbox.append(bbox[6])
				encoded_bboxs.append(encoded_bbox)
			batch_bbox.append(encoded_bboxs)
		return torch.tensor(batch_bbox).to(xyz_tensor.device.type)