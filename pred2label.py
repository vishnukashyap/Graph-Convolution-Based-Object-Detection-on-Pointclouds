import torch
import numpy as np
import json

reg_output = torch.load("path to saved file").squeeze(0).numpy().tolist()

preds_json = {}
preds_json['bounding boxes'] = []

idx = 0

for i in range(0,len(reg_output)):

	box = {}
	h,w,l,x,y,z,r = reg_output[i]
	box['center'] = {'x':x,'y':y,'z':z}
	box['height'] = h
	box['width'] = w
	box['length'] = l
	box['angle'] = r
	preds_json['bounding boxes'].append(box)

	if i % 20 == 0:
		file = open('results/temp'+str(idx)+'.json','w')
		json.dump(preds_json,file)
		file.close()
		idx += 1
		preds_json['bounding boxes'] = []

file = open('results/temp'+str(idx)+'.json','w')
json.dump(preds_json,file)
file.close()
