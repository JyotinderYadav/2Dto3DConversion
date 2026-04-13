#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch

import numpy as np
import SimpleITK as sitk

from datetime import datetime 
from scipy.ndimage import zoom

from all_models import *


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Test Volume Path

# In[ ]:


VOLUME = './midas_mri.nrrd' # Path to volume


# In[ ]:


PATH_GENERATOR_WEIGHTS = '/home/aradhya/cycleGAN/32dim_new/bce_fcn/2020-03-03 22:29:54.884919_masked_cryo_final/models/150_0_g_.pt'


# ### Resize volume to 128^3

# In[ ]:


full_vol = sitk.ReadImage(VOLUME)
full_vol_np_ = sitk.GetArrayFromImage(full_vol)
full_vol_np_ = zoom(full_vol_np_, (128/full_vol_np_.shape[0],128/full_vol_np_.shape[1],88/full_vol_np_.shape[2]))


# ### Luminance Remapping with original visible human MRI volume

# In[ ]:


full_vol_np_ = (full_vol_np_ - full_vol_np_.min()) / (full_vol_np_.max() - full_vol_np_.min())
temp_vol_np = sitk.GetArrayFromImage(sitk.ReadImage('./volume_generator/resampled_16bit_128_128_88_registered_masked.mhd'))
full_vol_np_ = np.std(temp_vol_np) / np.std(full_vol_np_) * (full_vol_np_ - np.mean(full_vol_np_)) + np.mean(temp_vol_np)


# In[ ]:


full_vol_tensor = torch.from_numpy(full_vol_np_).unsqueeze(0).unsqueeze(0).float()


# # Load Generator

# In[ ]:


generator = FCN3D().to(device)
generator.load_state_dict(torch.load(PATH_GENERATOR_WEIGHTS))


# # Generate Poisson Grey Volume

# In[ ]:


if not os.path.exists('TEST_DIR'):
    os.makedirs('TEST_DIR')

now = str(datetime.now()) + '_16_out'
if not os.path.exists(os.path.join('TEST_DIR', now)):
    os.makedirs(os.path.join('TEST_DIR', now)) 


# In[ ]:


out = generator(full_vol_tensor.to(device))


# In[ ]:


out_np = out.squeeze(0).squeeze(0).detach().cpu().numpy()
out_sitk = sitk.GetImageFromArray(out_np)


# In[ ]:


in_np = full_vol_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
in_sitk = sitk.GetImageFromArray(in_np)


# # Save Volume

# In[ ]:


sitk.WriteImage(out_sitk, '{}/generated_16_temp.mhd'.format(os.path.join('TEST_DIR', now)))
sitk.WriteImage(in_sitk, '{}/mri_16in.mhd'.format(os.path.join('TEST_DIR', now)))


# In[ ]:





# In[ ]:




