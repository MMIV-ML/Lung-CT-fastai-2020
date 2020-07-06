import os.path
from image import *
import pandas as pd
from fastai.vision import *
from fastai.vision.data import SegmentationProcessor

class NiiImageList(fvision.ImageList): 
    
    def open(self, fn):
        "Open image in `fn`, subclass and overwrite for custom behavior."
        return open_nii_image(fn) 
    
    def reconstruct(self, t:fvision.Tensor): return NiiImage(t.float())
    
    def show_xys(self, xs, ys,  anatomical_plane:str=None, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
        rows = int(np.ceil(math.sqrt(len(xs))))
        axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)
        for x,y,ax in zip(xs, ys, axs.flatten()): x.show(ax=ax, y=y, anatomical_plane=anatomical_plane,  **kwargs)
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()
# -----------------------------------------------------------------------------------------------------------------
#Modified the code from: https://github.com/renato145/fastai_scans/blob/0285b806d3986bf2f1c12ec16ab6a0b905de6284/fastai_scans/data.py
def normalize_batch(x:fvision.TensorImage, **kwargs):
    mean = x.mean([2,3,4])[...,None,None,None]
    std = x.view(*x.shape[:2], -1).std(-1)[...,None,None,None]
    return (x-mean)/std

def normalize_funcs(b:Tuple[fvision.Tensor,fvision.Tensor], do_x:bool=True, do_y:bool=False):
    x,y = b
    if do_x: x = normalize_batch(x)
    if do_y: y = normalize_batch(y)
    return x,y

def denormalize(x:fvision.TensorImage): 
    return x.cpu()

class NiiDataBunch(fvision.DataBunch):

    def normalize(self, do_x:bool=True, do_y:bool=False):
        if getattr(self,'norm',False): raise Exception('Can not call normalize twice')
        self.norm = partial(normalize_funcs, do_x=do_x, do_y=do_y)
        self.denorm = denormalize
        self.add_tfm(self.norm)
        return self
    
    def show_batch(self, rows:int=5, ds_type:DatasetType=DatasetType.Train, reverse:bool=False, anatomical_plane:str=None, **kwargs)->None:
        '''
        Show a batch of data in `ds_type` on a few `rows`. 
        anatomical_plane: axial, coronal or sagittal
        '''
        x,y = self.one_batch(ds_type, True, True)
        if reverse: x,y = x.flip(0),y.flip(0)
        n_items = rows **2 if self.train_ds.x._square_show else rows
        if self.dl(ds_type).batch_size < n_items: n_items = self.dl(ds_type).batch_size
        xs = [self.train_ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
        #TODO: get rid of has_arg if possible
        if has_arg(self.train_ds.y.reconstruct, 'x'):
            ys = [self.train_ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
        else : ys = [self.train_ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
        self.train_ds.x.show_xys(xs, ys, anatomical_plane=anatomical_plane, **kwargs)
#---------------------------------------------------------------------------------------------------------
def normalize_data(data):
    data.__class__ = NiiDataBunch    
    return data.normalize()   
    

