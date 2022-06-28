import datetime
import numpy as np
import os
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import tqdm
import PointGCN
import dataset
import test
from loss import *


def train(dataset_dir,epochs,device,batch_size,checkpoint_dir,random_seed, writer_loss, train_step, val_step, last_ckpt=None,lr=0.01,momentum=0.9,validation_split=0.0):
	
	# Dataloader Initializer #
	kitti_dataset = dataset.kitti_dataset(dataset_dir)
	dataset_size = len(kitti_dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split*dataset_size))
	np.random.seed(random_seed)
	np.random.shuffle(indices)
	train_indices,validation_indices = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_indices)
	validation_sampler = SubsetRandomSampler(validation_indices)

	kitti_train_loader = torch.utils.data.DataLoader(kitti_dataset, batch_size=1, 
                                           sampler=train_sampler,num_workers=1)
	kitti_validation_loader = torch.utils.data.DataLoader(kitti_dataset, batch_size=1,
                                                sampler=validation_sampler,num_workers=1)

	# Model Initializer #
	no_classes = len(dataset.class_to_id.keys())
	model = PointGCN.PointGCN(True,no_classes).to(device)

	# Optimizer Initializer #
	optimizer = torch.optim.NAdam(model.parameters(),lr=lr)

	cur_epoch = 0
	train_loss = 0
	test_loss = 0
	cls_step,reg_step = 0,0

	if last_ckpt != None:
		model.load_state_dict(last_ckpt["model"])
		optimizer.load_state_dict(last_ckpt["optimizer"])
		cur_epoch = last_ckpt["epoch"]
		train_loss = last_ckpt["train_loss"]
		test_loss = last_ckpt["test_loss"]
		train_step = last_ckpt["tensorboard_train_step"]
		val_step = last_ckpt["tensorboard_val_step"]
		for groups in optimizer.param_groups:
			groups['lr'] = lr 
	
	for epoch in range(cur_epoch,epochs):
		pbar = tqdm.tqdm(enumerate(kitti_train_loader))
		pbar.set_description("Epoch: "+str(epoch)+" | Training Loss: "+str(train_loss))#+" | Testing Loss: " +str(test_loss))
		count = 0
		model.train()
		for batch_idx, (data,target,_) in pbar:
			data = data.to(device)
			target = target.to(device)

			if np.all(target.cpu().numpy() == torch.zeros(1,1,9).numpy()):
				continue

			pc,classification_output,regression_output = model(data)
			batch_loss = compute_batch_loss(target,pc,classification_output,regression_output,writer_loss,cls_step,reg_step)/batch_size

			cls_step += 1
			reg_step += 1
			train_loss += batch_loss.item()

			batch_loss.backward()

			count += 1
			pbar.set_description("Epoch: "+str(epoch)+" | Training Loss: "+str(train_loss))#+" | Testing Loss: " +str(test_loss))
			if count == batch_size or batch_idx==len(kitti_train_loader)-1 :
				writer_loss.add_scalar("Training loss", (train_loss), global_step=train_step)
				train_step += 1
				optimizer.step()
				optimizer.zero_grad()
				count = 0
				train_loss = 0
			del data
			del target
			del pc
			del classification_output
			del regression_output
			torch.cuda.empty_cache()

		if validation_split > 0:
			test_loss = test.compute_validation_loss(kitti_validation_loader,model,device)
			writer_loss.add_scalar("Testing loss", test_loss, global_step=val_step)
			val_step += 1

		# Saving the weights
		datetime_now = datetime.datetime.now()
		date_now = str(datetime_now.date()).replace("-","_")
		time_now = str(datetime_now.time()).replace(":","_").replace(".","_")
		checkpoint_file = "Checkpoint_"+date_now+"_"+time_now+"_Ckpt_"+str(epoch)+".pt"

		checkpoint = {}
		checkpoint["epoch"] = epoch
		checkpoint["model"] = model.state_dict()
		checkpoint["optimizer"] = optimizer.state_dict()
		checkpoint["train_loss"] = train_loss
		checkpoint["test_loss"] = test_loss
		checkpoint["tensorboard_train_step"] = train_step
		checkpoint["tensorboard_val_step"] = val_step
		save_dir = os.path.join(checkpoint_dir,checkpoint_file)
		torch.save(checkpoint,save_dir)

		del checkpoint
		torch.cuda.empty_cache()

def main():

	dataset_dir = "./dataset"
	epochs = 20
	device = 'cuda'
	batch_size = 16
	checkpoint_dir = "./weights"
	last_ckpt = None #torch.load("path to checkpoint")
	random_seed = 42
	writer_loss = SummaryWriter(f'logs/loss')
	train_step = 0
	val_step = 0
	validation_split = 0.0

	train(dataset_dir,epochs,device,batch_size,checkpoint_dir,random_seed, writer_loss, train_step, val_step, last_ckpt,lr=0.01,momentum=0.9,validation_split=0.0)
	writer_loss.close()

if __name__ == '__main__':
	main()