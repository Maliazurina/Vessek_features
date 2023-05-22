import os
import pandas as pd
import pathlib
import numpy as np
root_dir = pathlib.Path.cwd()
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize 
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from skimage.morphology import skeletonize, skeletonize_3d, remove_small_objects

# Supressing the warning messages
import warnings
warnings.filterwarnings('ignore')

#--------functions---------------    
def resize_1mm(path): 
    img = nib.load(path) 
    img_array = img.get_fdata()
    img_affine = img.affine
    pixel_dim = np.diag(np.abs(img_affine))[0:3]
    
    for i in range(3):
        img_affine[i][i] = img_affine[i][i]/abs(img_affine[i][i])
    
    img_3d = img_array.astype(np.float32)
    
    phy = img_3d.shape*pixel_dim    # physical size
    iso = 1 # isotropic voxel
    new_size = np.round(phy/iso)   # new resampling size after interpolation 
    img_3d_interp = resize(img_3d, (new_size[0],new_size[1],new_size[2]), order=1, preserve_range=True)
    new_pixel_dim = np.round(phy/new_size)
    '''if save_nii:
        array_interp_NIFTI = nib.Nifti1Image(vessel_3d_interp, vessel_seg_affine)
        array_interp_NIFTI.to_filename(save_nii)'''
    return img_3d_interp,img_affine
                
def inner_outer_boundary(ct, tumor_mask, lung_mask, img_resolution, itr_dist):
    tumor_mask_inner = np.empty(ct.shape)
    tumor_mask_outer = np.empty(ct.shape)
    tumor_mask_core = np.empty(ct.shape)
    
    iterate_num = np.round(itr_dist/img_resolution[0])
    
    # loop through the nonzero slices
    sum_vec = tumor_mask.sum(0).sum(0)
    sele_idx = np.nonzero(sum_vec)
    for k in sele_idx[0]:
        kernel = np.ones((3, 3), np.uint8)
        tumor_mask_2D = tumor_mask[:,:,k] * 1
        tumor_mask_2D = tumor_mask_2D.astype(np.uint8)
        mask_dist = distance_transform_edt(tumor_mask_2D)
        max_dist = np.amax(mask_dist)
        mask_erode = mask_dist > (max_dist/2)
        mask_erode = mask_erode * 1
        tumor_mask_core[:, :, k] = mask_erode
        tumor_mask_inner[:, :, k] = tumor_mask_2D - mask_erode
        mask_dist2 = distance_transform_edt(1-tumor_mask_2D)
        mask_dilate = mask_dist2 <= iterate_num
        mask_dilate = mask_dilate * 1
        tumor_mask_outer[:, :, k] = mask_dilate
        
    # refine the analysis to the area within lung parenchyma
    lung_mask = np.logical_or(tumor_mask, lung_mask) * 1
    tumor_mask_outer = np.logical_and(lung_mask, tumor_mask_outer) * 1
    tumor_mask_core = tumor_mask_core * 1
    tumor_mask_inner = tumor_mask_inner * 1
    
    
    return tumor_mask_core,tumor_mask_inner,tumor_mask_outer
    
    
#------------raw data---------------------------
root_dir = pathlib.Path.cwd()
raw_data = 'C:/Local_Malia/MDA_DATASET/Codes/SABR/WCLC/Raw_Data'
#raw_data = 'C:/Local_Malia/MDA_DATASET/Codes/SABR/Regional_Variation/Data'
ct_path = os.path.join(raw_data,'CT')
tumor_path = os.path.join(raw_data,'SAMR')
lung_path= os.path.join(raw_data,'SAML')    
itr_dist = 5

#---resampled_data
#ct_iso  = os.path.join(root_dir,'CT_isotropic')  
#os.makedirs(ct_iso,exist_ok=True)
'''
lung_iso  = os.path.join(root_dir,'Lung_isotropic')  
os.makedirs(lung_iso,exist_ok=True)
tumor_iso  = os.path.join(root_dir,'Tumor_isotropic')  
os.makedirs(tumor_iso,exist_ok=True)'''


#inner_ring  = os.path.join(root_dir,'inner_ring')  
#os.makedirs(inner_ring,exist_ok=True)
outer_ring  = os.path.join(root_dir,'sam_outer_ring_5mm')  
os.makedirs(outer_ring,exist_ok=True)

image_names = sorted([ele for ele in os.listdir(ct_path) if ele.endswith(".nii.gz")])
image_paths = [os.path.join(ct_path, ele) for ele in image_names]


for ind, cur_img_path in enumerate(image_paths):
    file_name = os.path.basename(cur_img_path).split('_', 1)[1]
    print("Resampling {} {:3d}/{:3d}".format(file_name, ind+1, len(image_paths)))
    
    cur_ct_path = cur_img_path
    cur_lung_path = os.path.join(lung_path,'SAML_' + file_name)
    cur_tumor_path = os.path.join(tumor_path,'SAMR_' + file_name)
    
    ct_isotropic,img_affine = resize_1mm(cur_ct_path)
    #array_interp_NIFTI = nib.Nifti1Image(ct_isotropic, img_affine)
    #filename_nifti = os.path.join(ct_iso, 'CT_'+file_name)
    #array_interp_NIFTI.to_filename(filename_nifti)
    
    lung_isotropic,img_affine = resize_1mm(cur_lung_path)
    #array_interp_NIFTI = nib.Nifti1Image(lung_isotropic, img_affine)
    #filename_nifti = os.path.join(lung_iso, 'SAML_'+file_name)
    #array_interp_NIFTI.to_filename(filename_nifti)
    
    tumor_isotropic,img_affine = resize_1mm(cur_tumor_path)
    #array_interp_NIFTI = nib.Nifti1Image(tumor_isotropic, img_affine)
    #filename_nifti = os.path.join(tumor_iso, 'SAMR_'+file_name)
    #array_interp_NIFTI.to_filename(filename_nifti)
    
    img_resolution =[1,1,1]
    tumor_isotropic = tumor_isotropic.astype(np.bool)
    lung_isotropic = lung_isotropic.astype(np.bool)
    
    tumor_mask_core,tumor_mask_inner,tumor_mask_outer = inner_outer_boundary(ct_isotropic, tumor_isotropic, lung_isotropic, img_resolution, itr_dist = itr_dist)
    tumor_mask_core = tumor_mask_core.astype(np.uint8)
    tumor_mask_inner = tumor_mask_inner.astype(np.uint8)
    tumor_mask_outer = tumor_mask_outer.astype(np.uint8)
        
    tumor_mask_outer_nifti = nib.Nifti1Image(tumor_mask_outer, img_affine)
    filename_nifti = os.path.join(outer_ring, 'tumor_outer_'+file_name)
    tumor_mask_outer_nifti.to_filename(filename_nifti)




