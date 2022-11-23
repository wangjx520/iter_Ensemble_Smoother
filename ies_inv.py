import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
import os  

#%%
def get_l1loss(output,target):
	loss=abs(output-target).mean()
	return loss


def get_output_loss(model,ensemble,inv_target,in_feature,wbase):

	batch_size=200
	ensemble_outputs=[]
	for i in range(0,ensemble.shape[0],batch_size):
		ensemble_tensor=torch.FloatTensor(ensemble[i:i+batch_size,:]).reshape(-1,in_feature)
		ensemble_outputs.append(model(ensemble_tensor).detach().numpy().reshape(-1,len(inv_target)))  #集成点预测的

	ensemble_outputs=np.vstack(ensemble_outputs)
	ensemble_outputs=ensemble_outputs/np.tile(wbase,(ensemble_outputs.shape[0],1))

	abs_error=get_l1loss(ensemble_outputs[:-1,:],inv_target)
	print('optimize abs_error=',abs_error)

	return ensemble_outputs,abs_error

def inner_iteration(model,ies_args,ensemble,ensemble_outputs,perturbed_data,inv_targets,\
					deltaM,deltaD,lambd,wbase,nd,ne,nf,error,iterat,ud,wd,vd,svdpd):

	iter_lambd  =0
	max_inn_iter=ies_args['max_in_iter']
	lambd_incre =ies_args['lambd_incre']
	lambd_reduct=ies_args['lambd_reduct']
	min_rn      =ies_args['min_rn']
	is_min_rn=0
	
	
	while iter_lambd<max_inn_iter:
		
		#print('-----inner iteration step:',iter_lambd,'------')
		ensemble_old=ensemble.copy()
		ensemble_outputs_old=ensemble_outputs.copy()
		
		if ies_args['do_tsvd']:
			alpha=lambd*np.sum(wd**2)/svdpd
			x1=vd@sparse.diags(wd/(wd**2+alpha),0,(svdpd,svdpd))
			kgain=deltaM.T@x1@ud.T

		else:

			alpha=lambd*sum(sum(deltaD**2))/nd 
			kgain=deltaM.T@deltaD@np.linalg.inv(deltaD.T@deltaD+alpha*np.eye(nd))

		iterated_ensemble=ensemble[:ne,:]-(ensemble_outputs[:ne,:]-perturbed_data)@kgain.T 
		ensemble_mean=iterated_ensemble.mean(axis=0)
		ensemble=np.vstack([iterated_ensemble,ensemble_mean])

		m_change=np.sqrt(np.sum((ensemble[-1,:]-ensemble_old[-1,:])**2)/nf)
		#print('average change of ensemble mean=',m_change)

		ensemble_outputs,error_new=get_output_loss(model,ensemble,inv_targets,ies_args['in_feature'],wbase)
		#print('-----------','\n','abs_error_new=',error_new,'\n','-------------------')

		if error_new>error:
			lambd=lambd*lambd_incre
			print('lambd increase to',lambd)
			
			ensemble_outputs=ensemble_outputs_old
			ensemble=ensemble_old
			iter_lambd=iter_lambd+1

		else:
			lambd=lambd*lambd_reduct
			#print('lambd reduce to',lambd)
			iterat=iterat+1
			
			ensemble_outputs_old=ensemble_outputs
			ensemble_old=ensemble
	
			if abs(error_new-error)/abs(error)*100<min_rn:
				is_min_rn=1	
				
			error=error_new
			break

	return ensemble,ensemble_outputs,error,lambd,is_min_rn,iterat,iter_lambd

def outter_iteration(model,ies_args,ensemble,inv_target,perturbed_data,ensemble_outputs,\
			wbase,nd,ne,nf,init_error):
	
	error=init_error
	init_lambd  =ies_args['init_lambd']   
	lambd       =ies_args['init_lambd']
	lambd_incre =ies_args['lambd_incre']
	lambd_reduct=ies_args['lambd_reduct']
	max_lambd   =ies_args['max_lambd']


	tol_error   =ies_args['beta']**2*nd         # 容忍的误差
	min_rn      =ies_args['min_rn']

	do_tsvd =ies_args['do_tsvd']
	tsvd_cut=ies_args['tsvd_cut']

	max_out_iter=ies_args['max_out_iter']
	max_in_iter=ies_args['max_in_iter']

	iterat=0
	errors=[]
	lambds=[]
	
	ud,wd,vd,svdpd=0,0,0,0
	print('number of inv_target elements is',inv_target.size)
	while iterat<max_out_iter and error>tol_error:

		print('---------outer iteraion step:',iterat,'-------------')

		deltaM=ensemble[:ne,:]-np.ones((ne,1))@ensemble[ne:,:]
		deltaD=ensemble_outputs[:ne,:]-np.ones((ne,1))@ensemble_outputs[ne:,:]

		if do_tsvd:
			ud,wd,vd=np.linalg.svd(deltaD.T,full_matrices=False)
			vd=vd.T
			wd=np.diag(wd)
			val=np.diag(wd)
			total=np.sum(val)
			for j in range(1,ne):
				svdpd=j 
				if val[:j].sum()/total>tsvd_cut:
					break
			print('svdpd',svdpd)

			ud=ud[:,:svdpd]
			wd=val[:svdpd]
			vd=vd[:,:svdpd]


		ensemble,ensemble_outputs,error,lambd,is_min_rn,iterat,iter_lambd=inner_iteration(model,ies_args,\
			ensemble,ensemble_outputs,perturbed_data,inv_target,deltaM,deltaD,lambd,wbase,nd,ne,nf,error,iterat,\
			ud,wd,vd,svdpd)

		errors.append(error)
		lambds.append(lambd)

		if is_min_rn:
			print('终止外部循环，目标值的相对误差已经小于：',min_rn)
			#break
		
		if lambd>max_lambd:
			print('终止外部循环，lamda值已经大于：',max_lambd)
			break

		if iter_lambd>=max_in_iter:
			lambd=lambd*lambd_incre
			if lambd<init_lambd:
	 			lambd=init_lambd
			iterat=iterat+1
			print('终止内部循环，内部循环次数已经大于：',max_out_iter)
	
	if iterat>=max_out_iter:
		print('终止外部循环，iterat>=maxOuterIter')
	if error<=tol_error:
		print('终止外部循环，目标值已经小于：',tol_error)
	
	print('外部共循环:',iterat)
	return ensemble,errors,lambds


def ies_main(ies_args,target,model):
	model.eval()
	
	inv_target=target.flatten()                 # 目标值可以是多组
	principal_sqrtR=np.diag(np.ones(target.shape[-1]))  
	wbase=[]
	for i in range(target.shape[0]):  # obser=target
		wbase.append(np.diag(principal_sqrtR))
	wbase=np.array(wbase).flatten()
	inv_target=inv_target/wbase
	
	nd=len(inv_target)                          # 目标值的数量
	ne=ies_args['num_ensemble']                 # 集成的数量
	nf=ies_args['in_feature']*target.shape[0]   # in_feature*target[0]要反演的输入数据的个数
	
	#初始化集成的点，其中每行有target.shape[0]组初始化的输入数据
	ensemble=np.random.randn(ne,nf)  
	ensemble_mean=ensemble.mean(axis=0)[np.newaxis,:]
	ensemble=np.vstack([ensemble,ensemble_mean])

	# 获得随机初始化的ensemble通过正向模型得到的预测值ensemble_outputs和对应的绝对误差(目标值，目标是obj=0)
	ensemble_outputs,init_abs_error=get_output_loss(model,ensemble,inv_target,ies_args['in_feature'],wbase) 
	
	perturbed_data=np.zeros([ne,nd])       # 初始化集成点对应的输出值，在预测值的基础上进行随机扰动得到
	noise=ies_args['noise']*inv_target     # 扰动的边界值
	for i in range(ne):
		perturbed_data[i,:]=inv_target+noise*np.random.randn(*inv_target.shape)
		#perturbed_data[i,:]=measurement+noise*np.random.uniform(-1,1,measurement.size)

	#%%
	init_error=init_abs_error
	ensemble,errors,lambds=outter_iteration(model,ies_args,ensemble,inv_target,\
		perturbed_data,ensemble_outputs,wbase,nd,ne,nf,init_error)

	ensemble_output=model(torch.FloatTensor(ensemble[-1].reshape(-1,ies_args['in_feature']))).detach().numpy()
	erros=np.array(errors)
	lambds=np.array(lambds)

	return ensemble,ensemble_output,errors,lambds

