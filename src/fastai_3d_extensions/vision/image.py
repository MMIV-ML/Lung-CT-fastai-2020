#!/usr/bin/env python
# coding: utf-8

import sys 
import os.path

import numpy as np 
import matplotlib.pyplot as plt
import fastai.vision as fvision
from fastai.core import Category, FloatItem
from fastai.core import parallel
import torch
import nrrd
from monai.transforms import LoadNifti

def open_nii_image(fn): 
    x = None
    if str(fn).split('.')[-1] == 'nrrd': 
        _nrrd = nrrd.read(str(fn))
        x = _nrrd[0]
    else: 
        load_data = LoadNifti(image_only=True)
        x = load_data(fn) #nii file TODO aspect ratio
        
    if x is None: raise TypeError    
    return fvision.Image(torch.Tensor(x[None]))

def get_img_dimension(path, index): 
    return open_nii_image(path).shape[1:]

def get_largest_img_size(img_list, path=None, max_workers=20):
    if path: img_list = [f'{path}/{fn}' for fn in img_list]
    imgs_shape = np.array(parallel(get_img_dimension, img_list, max_workers=max_workers)).T
    return [max(imgs_shape[0]), max(imgs_shape[1]), max(imgs_shape[2])]

class NiiImage(fvision.ItemBase):
    "Support applying transforms to image data in `px`."
    def __init__(self, px:fvision.Tensor):
        self._px = px
        self._logit_px=None
        self._flow=None
        self._affine_mat=None
        self.sample_kwargs = {}
    
    def show(self, ax:plt.Axes=None, figsize:tuple=(3,3), hide_axis:bool=True,
              cmap:str=None, y:fvision.Any=None, anatomical_plane:str=None, **kwargs):
        "Show image on `ax`, using `cmap` if single-channel, overlaid with optional `y`"
        
        cmap = fvision.ifnone(cmap, fvision.defaults.cmap)

        if type(y) in [Category, FloatItem]: #classification or regression
            ax = show_image(self, ax=ax, hide_axis=hide_axis, figsize=figsize, anatomical_plane=anatomical_plane) 
            y.show(ax=ax, **kwargs)
        else: 
            ax = show_image(self, ax=ax, hide_axis=hide_axis, figsize=figsize, y=y,  anatomical_plane=anatomical_plane) #segmentation
            
    @property
    def data(self)->fvision.TensorImage:
        "Return this image pixels as a `LongTensor`."
        return self.px

    @property
    def px(self)->fvision.TensorImage:
        "Get the tensor pixel buffer."
        self.refresh()
        return self._px
    
    @px.setter
    def px(self,v:fvision.TensorImage)->None:
        "Set the pixel buffer to `v`."
        self._px=v

    def refresh(self)->None:
        "Apply any logit, flow, or affine transfers that have been sent to the `Image`."
        if self._logit_px is not None:
            self._px = self._logit_px.sigmoid_()
            self._logit_px = None
        if self._affine_mat is not None or self._flow is not None:
            self._px = _grid_sample(self._px, self.flow, **self.sample_kwargs)
            self.sample_kwargs = {}
            self._flow = None
        return self

def show_image(img:NiiImage, ax:plt.Axes=None, figsize:tuple=(3,3), hide_axis:bool=True, cmap:str='gray', cmap_y:str='jet', alpha:float=0.5, y=None, anatomical_plane:str=None, title:str=None, **kwargs)->plt.Axes:
    "Display `Image` in notebook."
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    show_slice_img(img=img, ax=ax, cmap=cmap, anatomical_plane=anatomical_plane, title=title, **kwargs)
    
    if y is not None: 
        show_slice_img(img=y, ax=ax, cmap=cmap_y, alpha=alpha, anatomical_plane=anatomical_plane, title=title, **kwargs) #TODO change cmap
        
    if hide_axis: ax.axis('off')
    return ax

def show_slice_img(ax, img, cmap, alpha:float=None, anatomical_plane:str=None, title:str=None, **kwargs):
    img_data = niiimage2np(img.data)
    slice_nr = img_data.shape[2] if anatomical_plane == 'sagittal' else img_data.shape[1] if anatomical_plane == 'coronal' else img_data.shape[0] #axial    
    slice_nr = slice_nr // 2
    
    img_data = img_data[:,:,slice_nr,0] if anatomical_plane == 'sagittal' else (img_data[:,slice_nr,:,0]) if anatomical_plane == 'coronal' else (img_data[slice_nr,:,:,0]) #axial
    ax.imshow(np.rot90(img_data), cmap=cmap, alpha=alpha, **kwargs)
    if title: ax.set_title(title)

    

def niiimage2np(image:fvision.Tensor)->np.ndarray:
    "Convert from torch style `image` to numpy/matplotlib style."
    res = image.cpu().permute(1,2,3,0).numpy()
    return res[...,0] if res.shape[2]==1 else res
