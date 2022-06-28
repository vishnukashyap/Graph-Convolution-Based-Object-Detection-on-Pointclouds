import torch
import tqdm
import PointGCN
import dataset
from loss import *
from iou import *
import time
import sys

def compute_validation_loss(dataloader,model,device):
	'''
		This function is will return the metrics for the validation set.
	'''
	val_loss = 0
	count = 0

	for batch_idx,(data,target,file_name) in enumerate(dataloader):
		data = data.to(device)
		target = target.to(device)

		pc,Z,M = test(model,data)
		torch.save(M.detach().cpu(),"./results/"+file_name[0][:-4]+"_Pred_bbox.pt")
		torch.save(Z.detach().cpu(),"./results/"+file_name[0][:-4]+"_Pred_class.pt")

		val_loss += compute_batch_loss(target,pc.to(device),Z.to(device),M.to(device),None,None,None).item()

		print(file_name)

		del data
		del target
		del pc
		del Z
		del M

	return val_loss/len(dataloader)


def test(model,data):
	'''
		This function returns the bboxes and their corresponding classification scores and the points they correspond to in the pointcloud
		This module is similar to the implementation of the box merging and scoring algorithm proposed in Point-GNN
	'''
	model.eval()

	IoU_thresh = 0.5
	classification_threshold = 0.7
	with torch.no_grad():
		pc,classification_output,regression_output = model(data)

		# here batch size will always be 1 so we can safely do cls_output[0] and reg_output[0] to get the predictions and bounding boxes 
		D = classification_output[0].clone().sigmoid()
		B = regression_output[0].clone()

		B[:,3:6] += pc[0]

		idx_of_d,B = get_final_pred_bbox(D,B)
		D = D[idx_of_d.type(torch.bool)]
		sampled_pc = pc[0,idx_of_d.type(torch.bool),:].clone()

		M = []
		Z = []
		final_PC = []
		idx_swap = torch.LongTensor([3,4,5,0,1,2,6]).cuda()

		while B.size()[0]>0:

			val, _ = torch.topk(D>classification_threshold,k=1,dim=1)
			_, i = torch.topk(val,k=1,dim=0)

			IoU3D = boxes_iou3d_gpu(B[:,idx_swap],B[i[0,0].data,:][idx_swap].unsqueeze(0)) # 1 x N
			boxes_selected = (IoU3D > IoU_thresh).squeeze(1)

			L = B[boxes_selected,:]
			selected_pc = sampled_pc[boxes_selected,:]

			if L.shape[0] == 0:
				B = torch.cat((B[0:i,:],B[i+1:,:]))
				D = torch.cat((D[0:i,:],D[i+1:,:]))
				sampled_pc = torch.cat((sampled_pc[0:i,:],sampled_pc[i+1:,:]))
				continue
			else:
				if not L.shape[0]%2 == 0:
					m = L[L.shape[0]//2,:]
					point_in_box = selected_pc[L.shape[0]//2,:]
				elif L.shape[0]==1:
					m = L[0,:]
					point_in_box = selected_pc[0,:]
				elif L.shape[0]%2==0:
					m = L[L.shape[0]//2,:]
					point_in_box = selected_pc[L.shape[0]//2,:]

				IoU3D = boxes_iou3d_gpu(m[idx_swap].unsqueeze(0),L)
				z = torch.sum(torch.mm(torch.clip(IoU3D,min=0),D[boxes_selected,:]),dim=0)

			if L.shape[0] > 0:
				B = B[~boxes_selected,:]
				D = D[~boxes_selected,:]
				sampled_pc = sampled_pc[~boxes_selected,:]

			M.append(m.unsqueeze(0))
			Z.append(z.unsqueeze(0))
			final_PC.append(point_in_box.unsqueeze(0))
		
		# torch.cuda.synchronize()
		M = torch.cat(M).unsqueeze(0).detach().cpu() # Predicted bounding boxes after NMS
		Z = torch.cat(Z).unsqueeze(0).detach().cpu() # Predicted classfication scores after NMS
		final_PC = torch.cat(final_PC).unsqueeze(0).detach().cpu() # Points corresponding to the predicted boxes after NMS

		if M.shape[1] == 0:
			return pc[0,idx_of_d.type(torch.bool),:],classification_output[idx_of_d.type(torch.bool)],regression_output[idx_of_d.type(torch.bool)]

		del classification_output
		del regression_output
		del D
		del B
		del val
		del i
		del L
		del IoU3D
		del m
		del z
		torch.cuda.empty_cache()
		
	return final_PC,Z,M

def get_final_pred_bbox(p_cls,p_reg):
	'''
		This function will predict the 
	'''

	_ , idx = torch.topk(p_cls,k=1,dim=1)
	
	classes = idx.squeeze(1)
	obj_bboxes = p_reg[classes.type(torch.bool)]

	return classes,obj_bboxes