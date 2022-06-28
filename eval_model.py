import torch
import tqdm

import dataset
import PointGCN
import test
import eval_map

def eval_model_accuracy(dataset_dir,device,batch_size,checkpoint):
	'''
		Evaluate the model's performance for the given trained weights 
	'''

	# Dataset and DataLoader Initialization #
	kitti_dataset = dataset.kitti_dataset(dataset_dir)
	kitti_loader = torch.utils.data.DataLoader(kitti_dataset, batch_size=1,num_workers=1)

	# Model Initializer #
	no_classes = len(dataset.class_to_id.keys())
	model = PointGCN.PointGCN(True,no_classes).to(device)
	model.load_state_dict(checkpoint["model"])

	mAP = 0.

	progress_bar = tqdm.tqdm(enumerate(kitti_loader))

	for batch_idx,(data,target,file_name) in progress_bar:
		
		progress_bar.set_description("Processing: "+file_name[0])

		data = data.to(device)
		target = target.to(device)

		_,pred_cls,pred_reg = test.test(model,data)

		mAP += eval_map.calculate_mAP(pred_reg[0].to(device),pred_cls[0].to(device),target[0])

	print("\n")
	print("The final calculated mAP for the current model is : " + str(mAP/len(kitti_dataset)))

	return

def main():

	dataset_dir = "./dataset"
	device = "cuda"
	batch_size = 8
	checkpoint = torch.load("weights/100pc_adam_30ep_linear/Checkpoint_2021_11_11_19_08_08_685077_Ckpt_29.pt")

	eval_model_accuracy(dataset_dir,device,batch_size,checkpoint)

if __name__ == '__main__':
	main()