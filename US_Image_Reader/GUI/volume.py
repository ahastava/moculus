import nibabel as nib
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cndimage
from scipy.ndimage import zoom
from cupyx.scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R

import time

class Volume:
    def __init__(self, file_path):
        self.load_volume(file_path)
        self.volume_gpu = cp.array(self.volume)
        self.rotated_gpu = self.volume_gpu.copy()
        self.x_dim, self.y_dim, self.z_dim = self.volume_gpu.shape
        print('volume shape: ', self.x_dim, self.y_dim, self.z_dim)

    def load_volume(self, file_path):
        nii_img = nib.load(file_path)
        self.volume = nii_img.get_fdata()
        self.volume_updated = self.volume.copy()

    def scale(self, method='GPU', scale_factor=0.2):
        if method == 'GPU':
            self.volume_gpu = cndimage.zoom(self.volume_gpu, zoom=scale_factor, order=0)

            self.rotated_gpu = self.volume_gpu.copy()
            self.x_dim, self.y_dim, self.z_dim = self.volume_gpu.shape
            print('volume shape: ', self.x_dim, self.y_dim, self.z_dim)
        else:
            self.volume_updated = zoom(self.volume, zoom=(scale_factor, scale_factor, scale_factor), order=3)

    def pad(self, method='GPU'):

        # stack won't help, will create bad artifacts
        # stack_times = self.x_dim//self.z_dim
        # if stack_times == 0:  # avoid divide by 0
        #     stack_times = 1
        # self.volume_gpu = cp.tile(self.volume_gpu, (1, 1, stack_times))
        # print('stacked times:', stack_times)

        max_dim = max(self.volume_gpu.shape)*1.4
        pad_width = [((max_dim - s) // 2, (max_dim - s + 1) // 2) for s in self.volume_gpu.shape]
        #pad_width = [(0,0),(0,0),(1,0)]
        #pad_width = [(0, 0), (0, 0), (30, 30)]
        #pad_width = [(0, 0), (0, 0), ((max_dim - self.z_dim) // 2, (max_dim - self.z_dim) // 2)]
        #a = cp.asnumpy(self.volume_gpu[:, :,0])

        if method == 'GPU':
            pad_width_gpu = tuple((cp.int32(p[0]), cp.int32(p[1])) for p in pad_width)

            self.volume_gpu = cp.pad(self.volume_gpu, pad_width_gpu, mode='constant', constant_values=0)
            #a = cp.asnumpy(self.volume_gpu[:, :, 1])

            self.rotated_gpu = self.volume_gpu.copy()
            self.x_dim, self.y_dim, self.z_dim = self.volume_gpu.shape
            self.gpu_to_cpu()
        else:
            self.volume_updated = np.pad(self.volume_updated, pad_width, mode='constant', constant_values=0)


    def rotate_volume(self, yaw, pitch, roll):
        """ Rotates a 3D volume using a single affine transformation on the GPU. """

       #  # Convert angles from degrees to radians
       #  r = R.from_euler('xyz', [roll, yaw, pitch], degrees=True)
       #  rotation_matrix = r.as_matrix()  # 3x3 rotation matrix
       #
       #  # Convert 3x3 to 4x4 affine transformation matrix (for homogeneous coordinates)
       #  affine_matrix = cp.eye(4)
       #  affine_matrix[:3, :3] = cp.asarray(rotation_matrix)  # Apply rotation
       # # affine_matrix[:3, 3] = cp.asarray(self.volume_gpu.shape) / 2  # Center the rotation
       #
       #  # Apply affine transformation (fast GPU-based rotation)
       #  self.rotated_gpu = affine_transform(self.volume_gpu, affine_matrix[:3, :3], offset=affine_matrix[:3, 3], mode='nearest')
       #
        # Get volume shape

        #start_time = time.time()

        dz, dy, dx = self.volume_gpu.shape
        center = cp.array([dz / 2, dy / 2, dx / 2])  # Compute center of the volume

        # Convert angles (degrees to radians) and compute 3D rotation matrix
        r = R.from_euler('xyz', [roll, yaw, pitch], degrees=True)
        rotation_matrix = cp.asarray(r.as_matrix())  # 3x3 rotation matrix

        # Compute the correct offset to maintain center alignment
        offset = center - rotation_matrix @ center

        # Apply affine transformation (GPU-accelerated rotation)
        self.rotated_gpu = affine_transform(self.volume_gpu, rotation_matrix, offset=offset, mode='nearest')

        # elapsed_time = time.time() - start_time
        # print(f"GPU affine_transform Elapsed time: {elapsed_time:.3f} seconds")

    def rotate(self, yaw, pitch, roll, z_index, method='GPU'):
        #self.rotated_gpu = cndimage.rotate(self.volume_gpu, angle, axes=axes, reshape=False, mode='nearest')

        #start_time = time.time()

        self.rotated_gpu = cndimage.rotate(self.volume_gpu, yaw, axes=(0, 1), reshape=False, mode='constant', cval=0.0, order=0)
        self.rotated_gpu = cndimage.rotate(self.rotated_gpu, pitch, axes=(0, 2), reshape=False, mode='constant', cval=0.0, order=0)
        self.rotated_gpu = cndimage.rotate(self.rotated_gpu, roll, axes=(1, 2), reshape=False, mode='constant', cval=0.0, order=0)

        # order = 0 => 0.081 seconds (x3)
        # order = 1 => 0.121 seconds (x3)
        # order = 2 => 0.756 seconds (x3)
        # order = 3 => 1.349 seconds (x3)
        # order = 4 => 2.261 seconds (x3)
        # order = 5 =>  3.417seconds (x3)

        # elapsed_time = time.time() - start_time
        # print(f"GPU rotate Elapsed time: {elapsed_time:.3f} seconds")
        # start_time = time.time()

        self.rotated_gpu = cp.clip(self.rotated_gpu, 0, None)  # Use CuPy's clip function


        # elapsed_time = time.time() - start_time
        # print(f"GPU clip Elapsed time: {elapsed_time:.3f} seconds")


    # GPU rotate  Elapsed time: 1.295  seconds
    # GPU clip Elapsed time: 0.000  seconds
    # GPU display  Elapsed  time: 0.000  seconds

        return

    def gpu_to_cpu(self):
        self.volume_updated = cp.asnumpy(self.volume_gpu)


