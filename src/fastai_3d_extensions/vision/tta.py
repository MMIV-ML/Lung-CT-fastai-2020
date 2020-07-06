import copy
import numpy as np
import fastai.vision as fvision


def _tta_only(learn, ds_type, rand_transform): 
    "Computes the outputs for augmented inputs for TTA"

    tta_learn = copy.deepcopy(learn)
        
    if ds_type == fvision.DatasetType.Valid: tta_learn.data.valid_ds.tfms = tta_learn.data.train_ds.tfms
    elif ds_type == fvision.DatasetType.Test: tta_learn.data.test.tfms = tta_learn.data.train_ds.tfms
    
    tta_preds_list = []
    for i in range(rand_transform):
        preds,_ = tta_learn.get_preds(ds_type)
        tta_preds_list.append(np.array(preds))
        
    return tta_preds_list

def TTA(learn, ds_type=fvision.DatasetType.Valid, rand_transform:int=4): 
    preds,y = learn.get_preds(ds_type) 
    
    preds_list = _tta_only(learn, ds_type, rand_transform)
    preds_list.append(np.array(preds))
    
    final_preds = np.sum(preds_list, axis=0)/len(preds_list)
    return np.array(final_preds), y