
from cellpose import dynamics
from tifffile import imread,imsave
import sys,os
import numpy as np
from tqdm.auto import tqdm
import joblib

device = None
use_gpu = False

if use_gpu:
    import torch
    device = torch.device('cuda')
    print("using gpu")
else:
    print("using cpu")
    
directory = sys.argv[1]
omni = False

mask_files = [x for x in os.listdir(directory) if 'masks.tif' in x]

def compute_and_save_flow(directory,filename):
    mask_filename = os.path.join(directory,filename)
    flow_filename = mask_filename.replace('masks','flows')
    mask = imread(mask_filename)
    labels, dist, heat, veci = dynamics.masks_to_flows(mask,use_gpu=use_gpu,device=device,omni=False)
    if omni:
        flow = np.concatenate((labels[np.newaxis,:,:], dist[np.newaxis,:,:], veci,
                                heat[np.newaxis,:,:]), axis=0).astype(np.float32)
    else:
        flow = np.concatenate((labels[np.newaxis,:,:], labels[np.newaxis,:,:]>0.5, veci), axis=0).astype(np.float32)
    imsave(flow_filename, flow)
    
_ = joblib.Parallel(n_jobs=8)(
    joblib.delayed(compute_and_save_flow)(directory,filename) for filename in tqdm(mask_files)
)
