import torch
from iou import *
import dataset

def calculate_mAP(predicted_bounding_boxes,predicted_classification_scores,targets):
	'''
		predicted_bounding_boxes => size : (N,7)
		predicted_classification_scores => size : (N,num_cls+1(bg))
		target_bounding_boxes => size : (M,num_cls + 1(bg) + 7)
	'''
	
	mAP = 0.

	num_classes = len(dataset.class_to_id.keys())
	idx_swap = torch.LongTensor([3,4,5,0,1,2,6]).cuda()
	IoU_thresh = 0.5

	_, box_classes = torch.topk(predicted_classification_scores,k=1,dim=1)
	_, gt_box_classes = torch.topk(targets[:,:num_classes],k=1,dim=1)

	for i in range(1,num_classes):
		
		box_cls = box_classes.squeeze(1)==(i)
		cls_scores = predicted_classification_scores[box_cls,(i)]
		pred_bboxes = predicted_bounding_boxes[box_cls,:]

		gt_cls = gt_box_classes.squeeze(1)==(i)
		gt_bbox = targets[gt_cls,num_classes:]

		true_positives = torch.zeros(predicted_classification_scores.shape[0])
		false_positives = torch.ones(predicted_classification_scores.shape[0])
		total_gt_boxes = gt_bbox.shape[0]

		if total_gt_boxes == 0:
			continue

		best_box = torch.tensor([])
		best_box_idx = torch.tensor([])
		for j in range(gt_bboxes.shape[0])
			IoUs = boxes_iou3d_gpu(pred_bboxes[:,idx_swap],gt_bbox[j,idx_swap]) # 1 x N
			top_box, top_box_idx = torch.topk(IoUs,k=1,dim=1)
			best_box_idx = torch.cat((best_box_idx,top_box_idx))
			best_box = torch.cat((best_box,top_box))

		true_positives[best_box_idx.squeeze(1)] = (best_box>IoU_thresh).type(torch.FloatTensor).squeeze(1)
		false_positives[best_box_idx.squeeze(1)] = (~(best_box>IoU_thresh)).type(torch.FloatTensor).squeeze(1)

		true_positive_cs = torch.cumsum(true_positives,dim=0)
		false_positive_cs = torch.cumsum(false_positives,dim=0)

		recalls = true_positive_cs / (total_gt_boxes + 1e-5)
		precisions = true_positive_cs / (true_positive_cs + false_positive_cs + 1e-5)
		precisions = torch.cat((torch.tensor([1]), precisions))
		recalls = torch.cat((torch.tensor([0]), recalls))

		mAP += torch.trapz(precisions,recalls)

		del box_cls
		del cls_scores
		del pred_bboxes
		del gt_cls
		del gt_bbox
		del true_positives
		del false_positives
		del total_gt_boxes
		del IoUs
		del best_box_idx
		del true_positive_cs
		del false_positive_cs
		del recalls
		del precisions
		torch.cuda.empty_cache()

	del idx_swap
	del IoU_thresh
	del box_classes
	del gt_box_classes

	return mAP/(num_classes-1)