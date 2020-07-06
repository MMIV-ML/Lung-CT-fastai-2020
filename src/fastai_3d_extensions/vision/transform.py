from fastai.basics import *
from fastai.vision.image import _get_default_args
import fastai.vision as fvision
from typing import Union

#--------------------------------------------------------------------------------------------------------------------
class Transform():
    "Utility class for adding probability and wrapping support to transform `func`."
    order=0
    def __init__(self, func:fvision.Callable, order:fvision.Optional[int]=None):
        if order is not None: self.order=order
        self.func = func
        self.func.__name__ = func.__name__[1:] #To remove the _ that begins every transform function.
        self.def_args = _get_default_args(func)
        self.params = copy(func.__annotations__)
        
    def __call__(self, *args:fvision.Any, p=1., is_random:bool=True, use_on_y:bool=True, **kwargs:fvision.Any):
        "Calc now if `args` passed; else create a transform called prob `p` if `random`."
        if args: return self.calc(*args, **kwargs)
        else: return RandTransform(self, kwargs=kwargs, is_random=is_random, use_on_y=use_on_y, p=p)
        
    def calc(self, x:fvision.Image, *args:fvision.Any, **kwargs:fvision.Any):
        "Apply to image `x`, wrapping it if necessary."
        return self.func(x, *args, **kwargs)

@dataclass
class RandTransform():
    "Wrap `Transform` to add randomized execution."
    tfm:Transform
    kwargs:dict
    p:float=1.0
    resolved:dict = field(default_factory=dict)
    do_run:bool = True
    is_random:bool = True
    use_on_y:bool = True
    def __post_init__(self): functools.update_wrapper(self, self.tfm)

    def resolve(self)->None:
        "Bind any random variables in the transform."
        if not self.is_random:
            self.resolved = {**self.tfm.def_args, **self.kwargs}
            return

        self.resolved = {}
        # for each param passed to tfm...
        for k,v in self.kwargs.items():
            # ...if it's annotated, call that fn...
            if k in self.tfm.params:
                rand_func = self.tfm.params[k]
                self.resolved[k] = rand_func(*listify(v))
            # ...otherwise use the value directly
            else: self.resolved[k] = v
        # use defaults for any args not filled in yet
        for k,v in self.tfm.def_args.items():
            if k not in self.resolved: self.resolved[k]=v
        # anything left over must be callable without params
        for k,v in self.tfm.params.items():
            if k not in self.resolved and k!='return': self.resolved[k]=v()

        self.do_run = rand_bool(self.p)

    @property
    def order(self): return self.tfm.order

    def __call__(self, x:fvision.Image, *args, **kwargs):
        "Randomly execute our tfm on `x`."
        return self.tfm(x, *args, **{**self.resolved, **kwargs}) if self.do_run else x

#<-------------------------------Transformations---------------------------------------------------------->

def get_resize_dimension(largest_img_size:list,rescale:int):   
    '''
    Pad largest image size so it is divisble by the rescale value
    return padded img size and resizing dimension. 
    '''
    if (rescale is None) or (rescale <1): return largest_img_size, None
    
    padding_dimmension = [s if (s % rescale == 0) else (s + rescale) - (s % rescale) for s in largest_img_size] 
    resizing_dimmension_f = np.divide(padding_dimmension,rescale).astype(np.float32)
    resizing_dimmension = list(resizing_dimmension_f.astype(int))
    
    return padding_dimmension, resizing_dimmension


def get_transforms(largest_img_size:Union[list,tuple], rescale:int=None, min_zoom:float=1., max_zoom:float=1., degrees:float=0, translate_range:Union[list,tuple]=[0,0,0],train_xtra_tfms=None, validation_xtra_tfms=None): #todo check min_zoom
    largest_img_size, resizing_dimension = get_resize_dimension(largest_img_size, rescale) #padded in order to resize 
        
    tfms_train = [padding(largest_img_size=largest_img_size)]
    
    if resizing_dimension: tfms_train.append(resizing(size_list=resizing_dimension))
        
    tfms_val = tfms_train.copy()
        
    if max_zoom >1: tfms_train.append(rand_zoom(scale=(min_zoom, max_zoom)))
    if degrees>0: tfms_train.append(rand_rotate(degrees=(-degrees,degrees)))
    
    if len(translate_range) == 3 and all(i >0 for i in translate_range): 
        tfms_train.append(rand_translate(range_x=(-translate_range[0],translate_range[0]), range_y=(-translate_range[1],translate_range[1]), range_z=(-translate_range[2],translate_range[2])))

        #       train                               valid
    #return (tfms_train + listify(train_xtra_tfms),tfms_val + listify(validation_xtra_tfms))
    return (tfms_train, tfms_val)



from scipy.ndimage.interpolation import rotate
from monai.transforms import Resize, Zoomd, Rotated, Affine, Rand3DElastic
import torch

@fvision.TfmPixel
def resizing(x, size_list=None):
    if(size_list):
        x =  torch.Tensor(x[None])
        x = torch.nn.functional.interpolate(x, size=(size_list[0],size_list[1],size_list[2]), mode='nearest')
        x = x[0,:,:,:,:]
    return x

@fvision.TfmPixel
def resize(x, size_list=None, mode='nearest'):
    if(size_list):
        resize_data = Resize(spatial_size=(size_list[0],size_list[1],size_list[2]), mode=mode)
        x = resize_data(x)
        print(x.shape)
    return x
        

@fvision.TfmPixel
def padding(x, largest_img_size=None): 
    '''
    Pad image to the same size as the largest image in the data set.
    return padded image
    '''
    diff = np.asarray(largest_img_size) - np.asarray(x[0,:,:,:].shape)
    if len(diff[diff>=0]) != 3: 
        raise Exception('Data set contains images with shape: {}'.format(x[0,:,:,:].shape))
        
    if diff.sum() > 0:
        x = x[0,:,:,:]
        constant_values = np.amin(np.array(x))
        pad_size = [[diff[0]//2]*2,[diff[1]//2]*2,[diff[2]//2]*2]

        if diff[0] % 2 != 0: pad_size[0][1] +=1 
        if diff[1] % 2 != 0: pad_size[1][1] +=1
        if diff[2] % 2 != 0: pad_size[2][1] +=1
            
        npad = ((pad_size[0][0], pad_size[0][1]), (pad_size[1][0], pad_size[1][1]), (pad_size[2][0], pad_size[2][1]))
        x = np.pad(x, pad_width=npad, mode='constant', constant_values=0.)#constant_values)
        x = torch.Tensor(x[None])
    return x

@fvision.TfmPixel
def _zoom(x, scale:uniform=1.0):
    zoom = Zoomd(keys=['image'],zoom=scale, order=0, mode='constant', cval=0, prefilter=True, use_gpu=False, keep_size=True)
    data_dict = {'image':x.data}
    zoom_dict = zoom(data_dict)
    x = zoom_dict['image'][0]

    x = torch.Tensor(x[None])
    return x

zoom = Transform(_zoom)

def rand_zoom(scale):
    "Randomized version of `zoom`."
    return zoom(scale=scale, p=0.5)

@fvision.TfmPixel
def _rotate(x,  degrees:uniform=10.0): 
    rotation = Rotated(keys=['image'], angle=degrees, spatial_axes=(0, 1), reshape=False, order=0, mode='constant', cval=0, prefilter=True)
    
    data_dict = {'image':x.data}
    rotation_dict = rotation(data_dict)
    
    x = rotation_dict['image'][0]
    return torch.Tensor(x[None])

rotate = Transform(_rotate)

def rand_rotate(degrees:uniform):
    "Randomized version of `rotate`."
    return rotate(degrees=degrees, p=0.5)


#TODO need to look at translation for segmentation 
#interpolation order, mode = 'bilinear'
@fvision.TfmPixel
def _translate(x, range_x:uniform=0., range_y:uniform=0., range_z:uniform=0.):
    translate = Affine(rotate_params=None,
                       shear_params=None,
                       translate_params=(range_x, range_y, range_z),
                       scale_params=None,
                       spatial_size=list(x.shape[1:]),
                       padding_mode='zeros')
    
    x = translate(x)
    return torch.Tensor(x)

translate = Transform(_translate)

def rand_translate(range_x:uniform, range_y:uniform, range_z:uniform): 
    'Randomized version of translate'
    return translate(range_x=range_x, range_y=range_y, range_z=range_z, p=0.5)