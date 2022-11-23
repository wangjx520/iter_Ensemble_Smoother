import sys
sys.path.append('..')
import os
import copy
import time
import pickle
from datetime import datetime
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn

from ies_inv import ies_main

# 		
class linear_model(nn.Module):
    def __init__(self,input_channels,hidden_channels,output_channels):
        super(linear_model,self).__init__()
        
        self.layers=nn.Sequential(nn.Linear(input_channels,hidden_channels),
                                 nn.Dropout(p=0.1),
                                 nn.ReLU(),
                                 nn.Linear(hidden_channels,output_channels))
        
        

    def forward(self,x):
        x=self.layers(x)
        return x


#%%

if __name__=="__main__":
	
	model=torch.load('model.pt').cpu()
	# 模型为一个DNN网络，拟合函数f=x**3+2*y**2-7*z，

	#B_F
	test_input=np.random.randn(60,3)
	test_target=test_input[:,0:1]**3+2*test_input[:,1:2]**2-7*test_input[:,2:]**1	
	
	#%% 参数反演部分
	#'''
	ies_args={}

	ies_args['num_ensemble']=100  # 集成的数量
	ies_args['init_lambd']=1       # 初始化lambda
	ies_args['beta']=0.02          
	ies_args['max_out_iter']=200
	ies_args['max_in_iter']=500
	ies_args['lambd_incre']=1.2
	ies_args['lambd_reduct']=0.9
	ies_args['do_tsvd']=0
	ies_args['tsvd_cut']=0.99  # 特征值
	ies_args['min_rn']=0.01
	ies_args['noise']=0.1      # 对集成的目标施加的噪声
	ies_args['max_lambd']=600
	ies_args['in_feature']=3   # 输入数据的特征数
	ies_args['inv_num']=10    # 反演的次数


	inv_inputs_list,inv_outputs_list=[],[]
	errors_list,lambds_list=[],[]
	t0=time.time()
	for i in range(ies_args['inv_num']):
		ensemble,ensemble_output,errors,lambds=ies_main(ies_args,test_target,model)
		
		inv_inputs=ensemble[-1,:].reshape(-1,ies_args['in_feature'])
		inv_outputs=ensemble_output
		
		inv_inputs_list.append(inv_inputs)
		inv_outputs_list.append(inv_outputs)
		errors_list.append(errors)
		lambds_list.append(lambds)
		
	inv_inputs=np.array(inv_inputs_list)
	inv_outputs=np.array(inv_outputs_list)
	errors=np.array(errors_list)
	lambds=np.array(lambds_list)

	print('消耗时间为：',time.time()-t0)
	
	#%%
	import matplotlib.pyplot as plt
	fig,axs = plt.subplots(2,2,figsize=(10,10),dpi=200)

	for i in range(inv_inputs.shape[0]):
		
		axs[0,0].plot(inv_inputs[i].flatten(),test_input.flatten(),'o')
		axs[0,1].plot(inv_outputs[i].flatten(),test_target.flatten(),'o')
		
	
		axs[1,0].plot(errors[i])
		axs[1,1].plot(lambds[i])
	axs[0,1].plot([ensemble_output.min(),ensemble_output.max()],[ensemble_output.min(),ensemble_output.max()])
	axs[0,0].plot([int(np.min([inv_inputs.min(),test_input.min()])-1),int(np.max([inv_inputs.max(),test_input.max()])+1)],\
			        [int(np.min([inv_inputs.min(),test_input.min()])-1),int(np.max([inv_inputs.max(),test_input.max()])+1)])
	#axs[0,0].set_xlim([int(np.min([inv_inputs.min(),test_input.min()])-1),int(np.max([inv_inputs.max(),test_input.max()])+1)])
	#axs[0,0].set_ylim([int(np.min([inv_inputs.min(),test_input.min()])-1),int(np.max([inv_inputs.max(),test_input.max()])+1)])
	axs[0,0].set_xlabel('inv_inputs')
	axs[0,0].set_ylabel('test_inputs')
	axs[0,1].set_xlabel('inv_outputs')
	axs[0,1].set_ylabel('test_targets')
	axs[1,0].set_xlabel('out_iter')
	axs[1,0].set_ylabel('inv_errors')
	axs[1,1].set_xlabel('out_iter')
	axs[1,1].set_ylabel('lambda')	
	
	plt.tight_layout()
	fig.tight_layout()
	plt.savefig('inv_vis')

