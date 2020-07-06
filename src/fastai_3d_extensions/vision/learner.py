import sys 
import os
import os.path
from pathlib import Path
import fastai.vision as fvision
import torch.nn as nn
import copy



def create_learner(data, model, run_parallel=True, mixed_precision=True, model_dir=Path(os.getcwd())/'..'/'models', **learn_kwargs)->fvision.Learner:
    if run_parallel: model = nn.DataParallel(model) 
        
    learn = fvision.Learner(data, model, **learn_kwargs)
    learn.model_dir = model_dir
    
    learn.to_fp16() if mixed_precision else learn.to_fp32()
    return learn
    
def save_unet_weights_expect_final_layers(learn, save_model_name:str): 
    learn = copy.copy(learn)
    
    if type(learn.model) == nn.DataParallel: learn.model.module.model[2] = None 
    else: learn.model.model[2] = None
        
    learn.save(save_model_name)
        
    