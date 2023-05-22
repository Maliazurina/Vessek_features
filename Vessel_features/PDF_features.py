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
import cv2
import cc3d
from sklearn.metrics import mean_squared_error

from math import sqrt

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
    return img_3d_interp,img_affine
                


def distance_mapping(a,b,p=1):
    euc = sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))
    man = sum(abs(e1-e2) for e1, e2 in zip(a,b))
    hamm = sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)
    
    avg_dist = (euc+man+hamm)/3
    
    return euc,man,hamm,avg_dist
 
    
def pdf(ct,tumor,vessel):
    pixel_tumor = ct[tumor == 1]
    pixel_vessel = ct[vessel == 1]


    frq1, edges1 = np.histogram(pixel_tumor, bins=100, range=(-1000, 500)) 
    frq2, edges2 = np.histogram(pixel_vessel, bins=100, range=(-1000, 500)) 

    
    # normalize to pdf
    frq1 = frq1 / (np.sum(frq1) + 1e-10)
    frq2 = frq2 / (np.sum(frq2) + 1e-10)
    
    euc,man,hamm,avg_distance = distance_mapping(frq1,frq2,p=1)
    #norm_distance = abs(frq1-frq2)/avg_distance
    MSE = mean_squared_error(frq1, frq2)
    error = np.sqrt(MSE)
    
    f_hist_new = [euc,man,hamm,avg_distance,error]
    
    frq1 = frq1/avg_distance
    frq2 = frq2/avg_distance
    
    
    f_hist_compare1 = [
        cv2.compareHist(frq2.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_CORREL),
        cv2.compareHist(frq2.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_CHISQR),
        cv2.compareHist(frq2.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(frq2.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_BHATTACHARYYA)]
    
    
    return f_hist_compare1 + f_hist_new



    
#------------raw data---------------------------
root_dir = pathlib.Path.cwd()
raw_data = 'C:/Local_Malia/MDA_DATASET/Codes/SABR/WCLC/Raw_Data'
ct_path = os.path.join(raw_data,'CT_lung')
vessel_path = os.path.join(root_dir,'Mori_data','Vessel_iso')   # you put your RTS iso in nn folder only
outer_path  = os.path.join(root_dir,'Mori_data','SAM','sam_outer_10mm') 
tumor_path  = os.path.join(root_dir,'Mori_data','RTS_iso') # you put your RTS iso in nn folder only


image_names = sorted([ele for ele in os.listdir(ct_path) if ele.endswith(".nii.gz")])
image_paths = [os.path.join(ct_path, ele) for ele in image_names]

col_list = ['PDF' + str(x) for x in range(1,10)]
headers = None
row,patient_id = [],[]
feature_file = os.path.join(root_dir,'sam_outer_pdf_10mm.csv')
    
    
for ind, cur_img_path in enumerate(image_paths):
    file_name = os.path.basename(cur_img_path).split('_', 1)[1]
    pid = file_name.split('.', 1)[0]
    #tumor_name = os.path.join('tumor_outer_' + pid + '.nii.gz')
    outer_name = os.path.join('CT_' + pid + '_tumor_outer_seg.nii.gz')
    outer_mask = os.path.join(outer_path,outer_name)
    outer_mask = nib.load(outer_mask).get_fdata().astype(np.uint8)
    
    tumor_name = os.path.join('RTS_' + pid + '.nii.gz')
    tumor_mask = os.path.join(tumor_path,tumor_name)
    tumor_mask = nib.load(tumor_mask).get_fdata().astype(np.uint8)
    
    tumor_mask = outer_mask + tumor_mask
    tumor_mask[tumor_mask>0] = 1

    
    vessel_name = os.path.join('VS_' + pid + '.nii.gz')
    vessel_mask = os.path.join(vessel_path,vessel_name)
    vessel_mask = nib.load(vessel_mask).get_fdata().astype(np.uint8)
    
    print("Resampling {} {:3d}/{:3d}".format(file_name, ind+1, len(image_paths)))
    
    '''---Prepare CT istropic----'''
    ct_isotropic,img_affine = resize_1mm(cur_img_path)
    '''---check how many lesions exist----'''
    tumor_out, N = cc3d.connected_components(tumor_mask, return_N=True)
    vol,idx = [],[]
    if N > 1 : # if more than 1 lesions exist
        for segid in range(1, N+1):
            extracted_image = tumor_out * (tumor_out == segid)
            voxel_count = np.count_nonzero(extracted_image)
            vol.append(voxel_count)
            idx.append(segid)
    else:
        extracted_image =  tumor_out
        voxel_count = np.count_nonzero(extracted_image)
    vol.append(voxel_count)
    idx.append(1)
            
            
    max_vol =  np.argmax(vol)       
    tumor_mask = tumor_out * (tumor_out == idx[max_vol])
    tumor_mask[tumor_mask>0] = 1
    roi = tumor_mask + vessel_mask
    overlap_vessel_region = roi>=2; # True False ( bool)
    
    tumor_mask = tumor_mask.astype(np.bool)
    
    f_3D_histogram1 = pdf(ct_isotropic,tumor_mask,overlap_vessel_region)
    patient_id.append(pid)
    row.append(f_3D_histogram1)
    
row = np.vstack(row)
patient_id = np.vstack(patient_id)
final_features = pd.DataFrame(row,columns=[col_list])
final_features['Patient_ID'] = patient_id
final_features.to_csv(feature_file)





