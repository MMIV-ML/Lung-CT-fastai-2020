from fastai.callbacks.hooks import *
from fastai.core import * 
from monai.transforms import NormalizeIntensity
import fastai.vision as fvision
import torch

def normalize_img(in_img):
    normalizer = NormalizeIntensity()
    return torch.Tensor(normalizer(in_img))
    
def interpolate_and_normalize(avg_acts, img_size:list): 
    avg_acts =  torch.Tensor(avg_acts[None,None])
    avg_acts = torch.nn.functional.interpolate(avg_acts, size=img_size, mode='trilinear')
    avg_acts = avg_acts[0,:,:,:,:]
    return normalize_img(np.asarray(avg_acts))    
    
def hooked_backward(m, xb, cat):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g

def get_data(data, idx, ds_type): 
    return data.train_ds[idx] if ds_type == fvision.DatasetType.Train else data.valid_ds[idx] if ds_type == fvision.DatasetType.Valid else data.test_ds[idx]
    
def get_heatmap_elements(learn, data, idx:int=0, ds_type=fvision.DatasetType.Valid, cuda=True): 
    x,y = get_data(data, idx, ds_type)
    m = learn.model.eval()
    xb = x.data[None] #Get item into a batch
    if cuda: xb = xb.cuda() #on GPU
    return m, xb, y
    
def generate_cam(m,xb,y): 
    hook_a,hook_g = hooked_backward(m, xb, y)
    acts  = hook_a.stored[0].cpu()
    return interpolate_and_normalize(acts.mean(0), xb.shape[2:])#, hook_g, acts
    
    
def generate_grad_cam(m,xb,y): 
    hook_a,hook_g = hooked_backward(m, xb, y)
    acts  = hook_a.stored[0].cpu()
    
    grad = hook_g.stored[0][0].cpu()
    grad_chan = grad.mean(1).mean(1)
    grad_chan = grad_chan[...,None,None]
    grad_chan =  np.transpose(grad_chan, (0, 2, 3,1))
    mult = (acts*grad_chan).mean(0)
    
    return  interpolate_and_normalize(mult, xb.shape[2:])