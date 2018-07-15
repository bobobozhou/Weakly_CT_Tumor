from denseinference import CRFProcessor
import numpy as np
from scipy.io import loadmat
import os
from utilities import *

# generate test data
vol_name = '../../Data/public_data/volume/R01-001_vol_241.mat'
vol = np.array(loadmat(vol_name)['vol_patch_tumor'], dtype=float)
vol = (vol - vol.min())/(vol.max()-vol.min())

pos = np.repeat(vol[:, :, :, np.newaxis], 5, axis=3)

# init wrapper object
pro = CRFProcessor.CRF3DProcessor(max_iterations=10,
                                  pos_x_std=3.0,
                                  pos_y_std=3.0,
                                  pos_z_std=3.0,
                                  pos_w=3.0,
                                  bilateral_x_std=60.0,
                                  bilateral_y_std=60.0,
                                  bilateral_z_std=60.0,
                                  bilateral_intensity_std=20.0,
                                  bilateral_w=10.0,
                                  dynamic_z=False,
                                  ignore_memory=False,
                                  verbose=False)

# Now run crf and get hard labeled result tensor:
res = pro.set_data_and_run(vol, pos)

# Save Visualization
# plot_data_3d(vol, savepath=os.getcwd() + '/vol.png')
# plot_data_3d(res, savepath=os.getcwd() + '/res.png')
plot_data_cam_3d(vol, res, savepath=os.getcwd() + '/cam.png')