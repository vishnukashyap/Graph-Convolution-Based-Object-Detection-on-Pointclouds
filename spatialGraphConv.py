import torch

class SpatialGraphConv(torch.nn.Module):
	def __init__(self,channels_list,bn:bool = True):
		'''
		Spatial Graph Convolution as described in the paper named spatial graph convolutional networks.
		Input:
			in_channels (int): Here the number of channels is after excluding the 3 position vectors.
			out_channels (int): Just like above the output is channels are after excluding 3 position vectors.
		Note: The number of kernels can also be defined in which case there will be that many intermediate_layers which will be concatenated and passed to the final layer.
		'''
		super(SpatialGraphConv,self).__init__()
		self.in_channels = channels_list[0]
		self.out_channels = channels_list[-1]
		self.channels_list = channels_list
		self.intermediate_layer = torch.nn.Linear(3,self.in_channels) # 3 here indicates the x,y,z since the position of the points are considered during spatial convolution
		self.activation =  torch.nn.ReLU()
		feature_layers = []
		self.bn = bn
		for i in range(1,len(self.channels_list)):
			feature_layers.append(torch.nn.Linear(self.channels_list[i-1],self.channels_list[i],bias=True))
			if bn:
				feature_layers.append(torch.nn.BatchNorm1d(self.channels_list[i]))
			feature_layers.append(torch.nn.ReLU())
		self.feature_layers = torch.nn.Sequential(*feature_layers)
		self.residual_layers = torch.nn.Sequential(*[torch.nn.Conv1d(self.channels_list[0],self.channels_list[-1],1,bias=True),torch.nn.BatchNorm1d(self.channels_list[-1]),torch.nn.ReLU()])
		

	def forward(self,pointcloud_xyz,pointcloud_features,neighbourhood_pointclouds):
		'''
		Input:
			pointcloud_xyz: (batch_size,npoint,3)
			pointcloud_features: (batch_size,C,npoint)
			neighbourhood_pointclouds: (batch_size,3+C,npoint,nsample)
		Output:
			output (torch Tensor): This tensor contains the point cloud after graph convolution is applied . Its dimensions are (N,3+C_out), the position is maintained.
		'''

		pos_vector = neighbourhood_pointclouds.permute(0,2,3,1)[:,:,:,:3] - pointcloud_xyz.unsqueeze(2).repeat(1,1,neighbourhood_pointclouds.size()[-1],1)
		batch_output = self.feature_layers(torch.sum(self.activation(self.intermediate_layer(pos_vector))*neighbourhood_pointclouds.permute(0,2,3,1)[:,:,:,3:],dim=2).squeeze(0)).unsqueeze(0).permute(0,2,1)  + self.residual_layers(pointcloud_features)
			
		del pos_vector
		torch.cuda.empty_cache()

		return batch_output

class ResBlock(torch.nn.Module):
	def __init__(self,in_channels,out_channels,bn:bool=True):
		'''
			Residular Block 
		'''
		super(ResBlock,self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.Conv1d = torch.nn.Conv1d(self.in_channels,self.out_channels,1)
		self.bn = bn
		if bn:
			self.batch_norm = torch.nn.BatchNorm1d(self.out_channels)
		self.activation = torch.nn.ReLU()

	def forward(self,features):
		output = self.Conv1d(features)
		output = self.batch_norm(output)
		output = self.activation(output)

		return output



