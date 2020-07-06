"""
=========================================================
Preparing the LIDC/IDRI data
=========================================================
This script prepares the data downloaded from "Standardized 
representation of the TCIA LIDC-IDRI annotations using DICOM"
in the Cancer Imaging Archive.

(1) Files are unzipped
(2) For each subject, all the corresponding CT scans are 
    collected
(3) The CTs are normalized by clipping and squashing the range
(4) For each scan, all the nodules are collected
(5) For each nodule:
    (i)     Get all segmentation masks
    (ii)    Take their union (to ensure entire nodules are 
            captured)
    (iii)   Dilate the union masks slightly, to capture some 
            of their surroundings
    (iv)    Apply the resulting mask to the normalized CT 
            images, crop the results to the dilated masks, 
            and save the resulting nodule images as `nrrd`
            files


Parameters:
----------
path : str
    Absolute path to the downloaded zip files

dilation : int
    Amount of dilation. The default is dilation = 3.

/TODO/ _more to be added.


Notes
-----
v.July 2020: This first version of the script performs several 
unnecessary intermediate steps (for debugging purposes). 
A more flexible, faster and cleaner version will be uploaded later.

"""

# Written by A.S. Lundervold (allu@hvl.no) and S. Kaliyugarasan
# Date: 2020-07-06

import os
import numpy as np, pandas as pd
import argparse
import zipfile
from pathlib import Path
from glob import glob
import scipy.ndimage
import nrrd
import multiprocessing as mp
import pickle

#############################
# Convenience functions and settings
#############################

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

max_threads_to_use = mp.cpu_count()//2


#############################
# Parse arguments
#############################
parser = argparse.ArgumentParser()
parser.add_argument("DATA_DIR", type=str, 
                    help="Path to downloaded LIDC/IDRI data")

args = parser.parse_args()

DATA_DIR = Path(args.DATA_DIR)
# TODO: _More to be added here_
INTERIM_DATA = DATA_DIR/'interim'

# Store the DATA_DIR in a file, for use in later notebooks
save_object(DATA_DIR, 'path_to_raw_data.pkl')


#############################
# Unzip files
#############################

all_zips = list(DATA_DIR.glob('*.zip'))

## Check that data is downloaded
assert len(all_zips) == 31, (
        f"At least some zip files seem to be missing in {DATA_DIR}." 
        "Did you download all of them?")


print(f"Unzipping files in {DATA_DIR}. This will take a while (unless already done)...")


def unzip_file(fn):
    """
    Unzips a single archive to DATA_DIR
    if not already unzipped.
    """
    with zipfile.ZipFile(fn,"r") as zip_ref:
        for f in zip_ref.namelist():
            if os.path.exists(DATA_DIR/f'{f}') or os.path.isfile(DATA_DIR/f'{f}'):
                pass
            else:      
                zip_ref.extractall(DATA_DIR)   



pool = mp.Pool(max_threads_to_use) 
pool.map(unzip_file, all_zips, chunksize=1)
pool.close()


#############################
# Normalize and save CT scans
#############################

## Collect CT scans
all_subjects = sorted(list((DATA_DIR/'LIDC_conversion5').iterdir()))

def get_scans_of_subject(subj_path):
    """
    Input:
        subj_path: path to subject 
    Output:
        list of all the CT scans for the given subject
    """

    scans = sorted(list((subj_path.glob('*/*/*_CT*.*'))))  
    return scans


## Normalize
def normalize(subj_path, clip=(-1200., 600.), output_dir=INTERIM_DATA):
    """
    Move the CT intensity to range 'clip', then squash the range to [0, 1].
    Input:
        subj_path: path to a subject
        clip: clip range
    Output:
        for each CT scan of the subject, new (CT data, header) tuples saved as .nrrd
    """
      
    ct_scans = get_scans_of_subject(subj_path)
    
    subj_path = str(subj_path)
      
    for ct in ct_scans:
        ct = str(ct)
        output_dir = str(output_dir)
        output_scan_dir = output_dir + f'/{"/".join(ct.split("/")[-4:-1])}'
        output_scan_fn = output_scan_dir + '/' + ct.split("/")[-1].split(".")[0] + '-normalized.nrrd'
        
        if not os.path.isfile(output_scan_fn):

            try:
                ct_data, ct_header = nrrd.read(ct)  
                # Move range
                ct_data_new = (ct_data - clip[0])/(clip[1] - clip[0])

                # Squash to [0,1]
                ct_data_new[ct_data_new < 0] = 0
                ct_data_new[ct_data_new > 1] = 1
                ct_data_new = (ct_data_new*255).astype('uint8')

                # Save
                ## Create directories
                os.makedirs(output_scan_dir, exist_ok=True)
                ## Create nrrd
                nrrd.write(output_scan_fn, ct_data_new, ct_header, index_order='F')  
            except:
                print(f"Cannot process file {ct}")
                pass

        else:
            pass


## Run normalization

print(f"Normalizing the CT scans of {len(all_subjects)} subjects. "
        f"Storing results in {INTERIM_DATA} "
        "This will take a while...")

pool = mp.Pool(max_threads_to_use) 
pool.map(normalize, all_subjects, chunksize=1)
pool.close()

print("Normalization process complete.")

## Collect results

all_normalized_ct = list(INTERIM_DATA.glob('*/*/*/*normalized*.*'))

#assert len(all_normalized_ct) == 992, (
#    f"Normalization failed. Expected 992 CTs as a result. Got {len(all_normalized_ct)}")



###################################
# Collect all nodule annotations
# Create new mask from their unions
# Save the resulting masks
###################################

def get_list_of_nodules(ct_fn):
    ct_fn = str(ct_fn)
    ct_path = "/".join(ct_fn.split("/")[:-1])
    nodules = sorted(glob(f'{ct_path}/Nodule * *.nrrd'))
    nodules = set([n.split("/")[-1].split(" - ")[0] for n in nodules])
    return sorted(list(nodules))

def get_all_nodule_masks_nrrds(ct_fn):
    ct_fn = str(ct_fn)
    ct_path = "/".join(ct_fn.split("/")[:-1])
    different_nodules = get_list_of_nodules(ct_fn)
    nodules = {}
    for i in range(len(different_nodules)):
        nodules[i+1] = sorted(glob(f'{ct_path}/{different_nodules[i]} *.nrrd'))
    return nodules
                   
def create_union_nodule_mask(list_of_nodule_masks):
    nods = [nrrd.read(n) for n in list_of_nodule_masks]
    nods_data = [n[0] for n in nods]
    
    # Create union by adding elements, then binarizing
    union_data = sum(nods_data)
    union_data = np.where(union_data > 0, 1, 0)
    
    # Add header by using the one from the first nodule. 
    # To be able to save back as .nrrd
    union_data = (union_data, nods[0][1])
    return union_data

def save_union_nodule_masks(subj_path, output_dir=INTERIM_DATA):
    """
    Create new .nrrd files for all the nodules for all the CT scans of the given subject
    """
    
    
    subj_id = str(subj_path).split("/")[-1]
    
    ct_scans = get_scans_of_subject(subj_path)
    
    for ct in ct_scans:
        all_nodule_masks = get_all_nodule_masks_nrrds(ct)

        for i in all_nodule_masks.keys():
            current_masks = all_nodule_masks[i]
            output_mask_dir = str(output_dir) + f'/{"/".join(current_masks[0].split("/")[-4:-1])}'
            output_mask_fn = output_mask_dir + '/' + current_masks[0].split("/")[-1].split(".")[0].split(" - ")[0] + '-union.nrrd'
            
            if not os.path.isfile(output_mask_fn):           
                u = create_union_nodule_mask(current_masks)          

                # Save
                os.makedirs(output_mask_dir, exist_ok=True)
                nrrd.write(output_mask_fn, u[0], u[1], index_order='F')

            else:
                pass

## Run on all subjects
print(f"Saving segmentation masks for all subjects. "
        "This will take a while...")

pool = mp.Pool(max_threads_to_use) 
pool.map(save_union_nodule_masks, all_subjects, chunksize=1)
pool.close()


## Collect the results
all_union_nodule_masks = list(INTERIM_DATA.glob('*/*/*/*union*.*'))
#assert len(all_normalized_ct) == 2614, (
#    f"Creation of union masks failed. Expected 2614 masks. Got {len(all_union_nodule_masks)}")


                
###################################
# Dilation of masks
###################################


def save_dilated_image(nodule_union_mask_fn, output_dir=INTERIM_DATA, iterations=3, save=True):
    """
    Create and save a dilated mask around a nodule 
    by adding a border of size iterations
    """
    
    nodule_union_mask_fn = str(nodule_union_mask_fn)

    # Save
    output_mask_dir = str(output_dir) + f'/{"/".join(nodule_union_mask_fn.split("/")[-4:-1])}'
    output_mask_fn = output_mask_dir + '/' + nodule_union_mask_fn.split("/")[-1].split(".")[0].split(" - ")[0] + f'-dilated-{iterations}.nrrd'
    
    if not os.path.isfile(output_mask_fn):
        nodule_union_mask_nrrd = nrrd.read(nodule_union_mask_fn)
        n_data = nodule_union_mask_nrrd[0]
        n_header = nodule_union_mask_nrrd[1]
        struct = scipy.ndimage.generate_binary_structure(3, 3)

        dilated_data = scipy.ndimage.binary_dilation(n_data, struct, iterations=iterations).astype(n_data.dtype)
        dilated = (dilated_data, n_header)
        # Save
        os.makedirs(output_mask_dir, exist_ok=True)
        nrrd.write(output_mask_fn, dilated[0], dilated[1], index_order='F')
    else:
        pass



## Run on all subjects
print(f"Saving dilated masks for all nodules. "
        "This will take a while...")
pool = mp.Pool(max_threads_to_use) 
pool.map(save_dilated_image, all_union_nodule_masks, chunksize=1)
pool.close()

print(f"Creation of dilated masks done")

## Collect the results
all_dilated_nodule_masks = list(INTERIM_DATA.glob('*/*/*/*dilated*.*'))
#assert len(all_dilated_nodule_masks) == 2637, (
#    f"Creation of dilated masks failed. Expected 2637 masks. Got {len(all_dilated_nodule_masks)}")

                
###################################
# Apply masks to normalized CTs
# Save the results
###################################

def get_all_dilated_nodule_masks(norm_ct_fn):
    norm_ct_fn = str(norm_ct_fn)
    ct_path = "/".join(norm_ct_fn.split("/")[:-1])
    
    nodules = sorted(glob(ct_path + '/*dilated*'))
    return nodules

def apply_mask(ct_fn, mask_fn):
    """
    Apply the mask to the CT image 
    """

    ct = nrrd.read(ct_fn)
    mask = nrrd.read(mask_fn)

    result_data = np.where(mask[0]>0, ct[0], 0)

    return (result_data, ct[1])

def crop_image(img,tol=0):
    """
    Crop the given image at the provided 
    tolerance level. 
    """
    mask = img>tol
    if img.ndim==3:
        mask = mask.any(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    return img[np.ix_(mask0,mask1)]

def crop_ct_and_save(ct_fn, mask_fn, crop=True, output_dir=INTERIM_DATA):
    """
    Apply the mask to the CT image and save the result
    """
    
    ct_fn = str(ct_fn)

    # Save
    output_dir = str(output_dir) + f'/{"/".join(ct_fn.split("/")[-4:-1])}'
    if crop: 
        output_fn = output_dir + '/CT-' + mask_fn.split("/")[-1].split(".")[0].split(" - ")[0] + f'-masked-cropped.nrrd'
    else:
        output_fn = output_dir + '/CT-' + mask_fn.split("/")[-1].split(".")[0].split(" - ")[0] + f'-masked.nrrd'
    
    if not os.path.isfile(output_fn):
        result_data, header = apply_mask(ct_fn, mask_fn)
        # Crop
        if crop:
            d = result_data
            xs,ys,zs = np.where(d!=0) 
            result_data = d[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1] 
        
        os.makedirs(output_dir, exist_ok=True)
        nrrd.write(output_fn, result_data, header, index_order='F')
    else:
        pass


## Run on all

print(f"Constructing final images. This will take a while...")

for ct in all_normalized_ct:
    for m in get_all_dilated_nodule_masks(ct):
        try:
            crop_ct_and_save(ct, m)
        except Exception as e:
            print(f"Error processing {ct} with mask {m}")
            print(e)
            print("#"*40)