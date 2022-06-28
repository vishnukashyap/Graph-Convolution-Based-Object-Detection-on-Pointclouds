import torch
import dataset
import numpy as np
import time
import iou

class FocalLoss(torch.nn.Module):
	'''
		Source: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
	'''
	def __init__(self, alpha=1, gamma=1.5, logits=False, reduce=True,reduction='none'):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.logits = logits
		self.reduce = reduce
		self.BCE_loss_criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

	def forward(self, inputs, targets):
		if self.logits:
			BCE_loss = self.BCE_loss_criterion(inputs, targets)
		else:
			BCE_loss = torch.nn.BCE(inputs, targets, reduce=False)

		# pred_prob = torch.sigmoid(inputs)
		# p_t = targets*pred_prob + (1-targets)*(1-pred_prob)
		# alpha_factor = targets*self.alpha + (1-targets)*(1-self.alpha)
		# modulating_factor = (1.0 - p_t)**self.gamma
		# BCE_loss *= alpha_factor*modulating_factor

		# if self.reduce:
		# 	return torch.mean(BCE_loss)
		# else:
		# 	return BCE_loss
			
		pt = torch.exp(-BCE_loss)
		F_loss = self.alpha * ((1-pt)**self.gamma) * BCE_loss

		if self.reduce:
			return torch.mean(F_loss)
		else:
			return F_loss

def point_in_gt(xyz,gt_bbox):
	'''
		Check if point is in the ground truth box or not
	'''
	x_min,y_min,z_min,x_max,y_max,z_max = xyzhwl2xyzxyz(gt_bbox)

	if xyz[0] > x_min and xyz[0] < x_max:
		if xyz[1] > y_min and xyz[1] < y_max:
			if xyz[2] > z_min and xyz[2] < z_max:
				return True
	return False

def xyzhwl2xyzxyz(bbox):
	x = bbox[3]
	y = bbox[4]
	z = bbox[5]
	h = bbox[0]
	w = bbox[1]
	l = bbox[2]

	output = torch.empty(6)

	output[0] = x - l*0.5
	output[1] = y - w*0.5
	output[2] = z
	output[3] = x + l*0.5
	output[4] = y + w*0.5
	output[5] = z + h

	return output

def GIoU(pred_bbox,gt_bbox):
	'''
		In this function the GIoU loss function is implemented for 3D bounding boxes
		As for the angle, a trignometric function will have to be used to calculate the loss
	'''
	pred_bbox[:6] = xyzhwl2xyzxyz(pred_bbox[:6])
	gt_bbox[:6] = xyzhwl2xyzxyz(gt_bbox[:6])
	# GIoU Loss for corrdinates

	# Intersection Volume
	intersection_xmin = max(pred_bbox[0],gt_bbox[0])
	intersection_ymin = max(pred_bbox[1],gt_bbox[1])
	intersection_zmin = max(pred_bbox[2],gt_bbox[2])

	intersection_xmax = min(pred_bbox[3],gt_bbox[3])
	intersection_ymax = min(pred_bbox[4],gt_bbox[4])
	intersection_zmax = min(pred_bbox[5],gt_bbox[5])

	# if intersection_xmax<intersection_xmin or intersection_ymax<intersection_ymin or intersection_zmax<intersection_zmin:
	# 	intersecion_volume = abs((intersection_xmax-intersection_xmin)*(intersection_ymax-intersection_ymin)*(intersection_zmax-intersection_zmin))*1e-2
	# else:
	# 	intersecion_volume = abs((intersection_xmax-intersection_xmin)*(intersection_ymax-intersection_ymin)*(intersection_zmax-intersection_zmin))

	intersecion_volume = abs((intersection_xmax-intersection_xmin)*(intersection_ymax-intersection_ymin)*(intersection_zmax-intersection_zmin))

	# Union Volume
	pred_bbox_volume = (pred_bbox[3]-pred_bbox[0])*(pred_bbox[4]-pred_bbox[1])*(pred_bbox[5]-pred_bbox[2])
	gt_bbox_volume = (gt_bbox[3]-gt_bbox[0])*(gt_bbox[4]-gt_bbox[1])*(gt_bbox[5]-gt_bbox[2])

	union_volume = pred_bbox_volume + gt_bbox_volume - intersecion_volume

	# Volume of box bounding both boxes
	c_bbox_xmin = min(pred_bbox[0],gt_bbox[0])
	c_bbox_ymin = min(pred_bbox[1],gt_bbox[1])
	c_bbox_zmin = min(pred_bbox[2],gt_bbox[2])

	c_bbox_xmax = max(pred_bbox[3],gt_bbox[3])
	c_bbox_ymax = max(pred_bbox[4],gt_bbox[4])
	c_bbox_zmax = max(pred_bbox[5],gt_bbox[5])

	c_bbox_volume = (c_bbox_xmax-c_bbox_xmin)*(c_bbox_ymax-c_bbox_ymin)*(c_bbox_zmax-c_bbox_zmin)

	GIoU_loss = ((intersecion_volume/union_volume)-((c_bbox_volume-union_volume)/c_bbox_volume))

	# Orientation Loss
	orientation_loss = abs(gt_bbox[6] - pred_bbox[6])

	loss = 1 - GIoU_loss + orientation_loss  # => 1 - GIoU + orientation loss

	return loss, GIoU_loss

def ry_to_rz(ry):
	angle = -ry - np.pi / 2

	if angle >= np.pi:
		angle -= np.pi
	if angle < -np.pi:
		angle = 2*np.pi + angle

	return angle

def xyzhwl_to_cornerpoints(bbox):
	x = bbox[3]
	y = bbox[4]
	z = bbox[5]
	h = bbox[0]
	w = bbox[1]
	l = bbox[2]
	ry = bbox[6]

	Box = torch.tensor([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
					[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
					[0, 0, 0, 0, h, h, h, h]],dtype=torch.float64)

	rz = ry_to_rz(ry)

	rotMat = torch.tensor([
		[torch.cos(rz), -torch.sin(rz), 0.0],
		[torch.sin(rz), torch.cos(rz), 0.0],
		[0.0, 0.0, 1.0]],dtype=torch.float64)

	velo_box = torch.mm(rotMat, Box).to(bbox.device.type)

	# velo_box = torch.tensor(velo_box)
	velo_box[0] += x
	velo_box[1] += y
	velo_box[2] += z
	
	return velo_box

def mse_regression_loss_calculation(pbbox,gtbbox):
	pbbox = xyzhwl_to_cornerpoints(pbbox)
	gtbbox = xyzhwl_to_cornerpoints(gtbbox)

	reg_loss_criterion = torch.nn.MSELoss(reduction='mean')
	reg_loss = reg_loss_criterion(pbbox,gtbbox)
	return reg_loss

def huber_loss_calculation(pbbox,gtbbox):
	pbbox = xyzhwl_to_cornerpoints(pbbox)
	gtbbox = xyzhwl_to_cornerpoints(gtbbox)

	beta = 4.0
	reg_loss_criterion = torch.nn.SmoothL1Loss(beta=beta)
	reg_loss = reg_loss_criterion(pbbox,gtbbox)*beta
	return reg_loss

def compute_loss(xyz,p_cls,gt_cls,p_bbox,gt_bbox):
	'''
		The rehression loss used in the PointGNN paper is the Huber Loss
		Here we try using a 3D GIoU loss
		If it doesnt work huber loss will be used
	'''
	cls_loss_criterion = FocalLoss(alpha=1,gamma=3,logits=True,reduce=False,reduction='mean')
	p_bbox[3:6] += xyz
	for gt in range(gt_bbox.size()[0]):
		if point_in_gt(xyz,gt_bbox[gt]):
			cls_loss = cls_loss_criterion(p_cls,gt_cls[gt])
			# reg_loss, _ = GIoU(p_bbox,gt_bbox[gt])
			# reg_loss = mse_regression_loss_calculation(p_bbox,gt_bbox[gt])
			reg_loss = huber_loss_calculation(p_bbox,gt_bbox[gt])
			break
		else:
			target = gt_cls[gt]*0.
			target[0] = 1
			cls_loss = cls_loss_criterion(p_cls,target)
			# reg_loss, _ = GIoU(p_bbox,gt_bbox[gt])
			# reg_loss = mse_regression_loss_calculation(p_bbox,gt_bbox[gt])
			reg_loss = huber_loss_calculation(p_bbox,gt_bbox[gt])
			reg_loss = reg_loss*1e-6
	# total_loss = cls_loss+reg_loss
	# return total_loss
	return cls_loss,reg_loss

def total_loss_calculation_all_bboxes(xyz,pred_bboxes,gt_bboxes,pred_cls,gt_cls):
	'''
		Check which all points are in the bounding boxes
	'''
	beta = 1.0
	reg_loss_criterion = torch.nn.SmoothL1Loss(beta=beta,reduction='none')
	cls_loss_criterion = FocalLoss(alpha=1,gamma=1.5,logits=True,reduce=False)

	pbboxes = pred_bboxes.clone()
	pbboxes[:,3:6] += xyz
	total_regression_loss = None
	total_classification_loss = None

	non_gt_bbox_pts = None

	for i in range(gt_bboxes.size()[0]):

		x_min,y_min,z_min,x_max,y_max,z_max = xyzhwl2xyzxyz(gt_bboxes[i])
		
		temp_x = torch.logical_and((xyz[:,0]<x_max),(xyz[:,0]>x_min))
		temp_y = torch.logical_and((xyz[:,1]<y_max),(xyz[:,1]>y_min))
		temp_z = torch.logical_and((xyz[:,2]<z_max),(xyz[:,2]>z_min)) 
		temp_output = torch.logical_and(temp_x,temp_y)
		temp_output = torch.logical_and(temp_output,temp_z)

		if non_gt_bbox_pts == None:
			non_gt_bbox_pts = temp_output
		else:
			non_gt_bbox_pts = torch.logical_or(non_gt_bbox_pts,temp_output)

		gt_reg_loss = torch.mean((torch.sum(reg_loss_criterion(pbboxes,gt_bboxes[i].repeat(pbboxes.size()[0],1)),dim=1)).reshape(-1,1)*temp_output,dtype=torch.float32)
		gt_cls_loss = 50.*torch.mean((torch.sum(cls_loss_criterion(pred_cls,gt_cls[i].repeat(pred_cls.size()[0],1)),dim=1)).reshape(-1,1)*temp_output,dtype=torch.float32)
		
		if total_regression_loss == None and total_classification_loss == None:
			total_regression_loss = gt_reg_loss
			total_classification_loss = gt_cls_loss
		else:
			total_regression_loss += gt_reg_loss
			total_classification_loss += gt_cls_loss

	background_points_target = torch.zeros(pred_cls.size()).cuda()
	background_points_target[:,0] = 1.
	non_gt_cls_loss = torch.mean((torch.sum(cls_loss_criterion(pred_cls,background_points_target),dim=1)).reshape(-1,1)*(~non_gt_bbox_pts),dtype=torch.float32)#/pbboxes.shape[0]
	if non_gt_cls_loss > 0.6:
		total_classification_loss += non_gt_cls_loss

	del background_points_target
	del temp_x
	del temp_output
	del temp_y
	del temp_z
	del pbboxes
	torch.cuda.empty_cache()

	return total_regression_loss, total_classification_loss

def compute_batch_loss(label,pointcloud,cls_output,reg_output,writer_loss,cls_step,reg_step):
	'''
		label : (b,no_targets,no_class+7)
		pointcloud: (b,npoint,3)
		cls_output: (b,npoint,no_class)
		reg_output: (b,npoint,7)
	'''	
	batch = label.size()[0]
	predictions = pointcloud.size()[1]
	no_class = len(dataset.class_to_id.keys())
	cls_target = label[:,:,:no_class]
	reg_target = label[:,:,no_class:]

	alpha = torch.tensor(4.0,dtype=torch.float32)
	beta = torch.tensor(1.0,dtype=torch.float32)
	
	# Bounding box loss
	for b in range(batch):
		pc_xyz = pointcloud[b,:,:]
		pred_cls = cls_output[b,:,:]
		pred_reg = reg_output[b,:,:]

		total_reg_loss,total_cls_loss = total_loss_calculation_all_bboxes(pc_xyz,pred_reg,reg_target[b],pred_cls,cls_target[b])
		# print("##################")
		# print(total_reg_loss)
		# print(total_cls_loss)
		# print("##################")
		# print("\nNew cls loss: ",total_cls_loss_)
		# print("New reg loss: ",total_reg_loss_)
		# for prediction in range(predictions):
		# 	gt_bbox = reg_target[b,:,:].clone().detach()
		# 	gt_cls = cls_target[b,:,:].clone().detach()
			
		# 	cls_loss,reg_loss = compute_loss(pc_xyz[prediction,:],pred_cls[prediction],gt_cls,pred_reg[prediction,:],gt_bbox)
		# 	if prediction==0:
		# 		total_cls_loss = cls_loss
		# 		total_reg_loss = reg_loss
		# 	else:
		# 		total_cls_loss += cls_loss
		# 		total_reg_loss += reg_loss
		# print("Old cls loss: ",total_cls_loss)
		# print("Old reg loss: ",total_reg_loss)
		if writer_loss is not None:
			writer_loss.add_scalar("Classification loss", total_cls_loss, global_step=cls_step)
			writer_loss.add_scalar("Regression loss", total_reg_loss, global_step=reg_step)
			# total_loss = compute_loss(pc_xyz[prediction,:],pred_cls[prediction],gt_cls,pred_reg[prediction,:],gt_bbox)
			# if prediction==0:
			# 	batch_loss=total_loss
			# else:
			# 	batch_loss += total_loss
		batch_loss = alpha*total_reg_loss + beta*total_cls_loss
	# del gt_bbox
	# del gt_cls
	del pred_cls
	del pred_reg
	del cls_target
	del reg_target
	del total_cls_loss
	del total_reg_loss
	torch.cuda.empty_cache()
	return batch_loss
