import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from numpy.random import randint
from deepinv.datasets import PatchDataset, generate_dataset
from deepinv.physics import Denoising
from deepinv.physics.noise import GaussianNoise
from torch.utils import data
from torch.utils.data import random_split
import h5py
from torch.utils.data import  DataLoader



class PatchesDataset(data.Dataset):
    r"""
    Builds the dataset of all patches from a tensor of images.

    :param torch.Tensor imgs: Tensor of images, size: batch size x channels x height x width
    :param int patch_size: size of patches
    :param callable: data augmentation. callable object, None for no augmentation.
    """

    def __init__(self, imgs, patch_size=6, stride=1, transforms=None):
        self.imgs = imgs
        self.patch_size = patch_size
        self.stride=stride
        self.patches_per_image_x = (self.imgs.shape[2] - patch_size) // stride + 1
        self.patches_per_image_y = (self.imgs.shape[3] - patch_size) // stride + 1
        self.patches_per_image = self.patches_per_image_x*self.patches_per_image_y
        self.transforms = transforms

    def __len__(self):

        return self.imgs.shape[0] * self.patches_per_image

    def __getitem__(self, idx):
        idx_img = idx // self.patches_per_image
        idx_in_img = idx % self.patches_per_image
        idx_x = (idx_in_img // self.patches_per_image_y) * self.stride
        idx_y = (idx_in_img % self.patches_per_image_y) * self.stride
        patch = self.imgs[
            idx_img, :, idx_x : idx_x + self.patch_size, idx_y : idx_y + self.patch_size
        ]
        if self.transforms:
            patch = self.transforms(patch)
        return patch, idx


def prepare_data(inputdata, patch_size, dir, physics, generator=None, stride=1, device="cpu", num_workers=1, batch_size=1):
    # scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                    transforms.RandomRotation([90, 90]), transforms.RandomRotation([180, 180]), 
                    transforms.RandomRotation([270, 270])])
    
    total_memory = torch.cuda.get_device_properties(device=device).total_memory
    memory_threshold = 0.6
    # Create an HDF5 file to store all patches
    h5_file_path = dir + '/all_patches.h5'
    with h5py.File(h5_file_path, 'w') as h5f:
        # Create expandable datasets for x_train, y_train, x_test, and y_test
        maxshape = (None, 3, patch_size, patch_size)  # None allows unlimited expansion on the first axis
        x_train_dataset = h5f.create_dataset("x_train", shape=(0, 3, patch_size, patch_size), maxshape=maxshape, dtype='float32', compression="gzip")
        y_train_dataset = h5f.create_dataset("y_train", shape=(0, 3, patch_size, patch_size), maxshape=maxshape, dtype='float32', compression="gzip")
        x_test_dataset = h5f.create_dataset("x_test", shape=(0, 3, patch_size, patch_size), maxshape=maxshape, dtype='float32', compression="gzip")
        y_test_dataset = h5f.create_dataset("y_test", shape=(0, 3, patch_size, patch_size), maxshape=maxshape, dtype='float32', compression="gzip")
        
        with torch.no_grad():
            for tensor_batch in inputdata:
                if(len(tensor_batch) > 1):
                    img,idx = tensor_batch
                else:
                    img = tensor_batch
                if not torch.is_tensor(img):
                    print('data should be tensor')
                    continue
                print(f"img shape: {img.shape}")
                b, c, h , w = img.shape
                for k in range(len(scales)):
                    trans = transforms.Resize([int(h*scales[k]), int(w*scales[k])])
                    Img = trans(img.detach())
                    
                    patches = PatchesDataset(Img, patch_size=patch_size, stride=stride, transforms=None)
                    
                    train_size = int(0.8 * len(patches))
                    test_size = len(patches) - train_size
                    print(f"patch+{idx}-shape{scales[k]}: patch_len{len(patches)}, train_size{train_size}, test_size{test_size}")
                    
                    if train_size == 0 or test_size == 0:
                        continue
                    
                    train_dataset, test_dataset = random_split(patches, [train_size, test_size])
                    dataloader_train = DataLoader(train_dataset, batch_size=len(patches), shuffle=True, num_workers=num_workers, pin_memory=True)
                    dataloader_test = DataLoader(test_dataset, batch_size=len(patches), shuffle=False, num_workers=num_workers, pin_memory=True)

                    # Stack patches from the datasets
                    all_train_data = []
                    all_train_target = []
                    for patch, _ in dataloader_train:
                        params = generator.step(batch_size=patch.shape[0])
                        patch_tr=transform(patch.detach().to(device))
                        all_train_data.append(patch_tr)
                        all_train_target.append(physics(patch_tr, **params))
                    
                    all_train_data = torch.cat(all_train_data, dim=0)
                    all_train_target = torch.cat(all_train_target, dim=0)
                    
                    all_test_data=[]
                    all_test_target=[]
                    for patch, _ in dataloader_test:
                        params = generator.step(batch_size=patch.shape[0])
                        patch_tr=patch.detach().to(device)
                        all_test_data.append(patch_tr)
                        all_test_target.append(physics(patch_tr, **params))

                    all_test_data = torch.cat(all_test_data, dim=0)
                    all_test_target = torch.cat(all_test_target, dim=0)

                    # Resize the datasets to accommodate new data
                    x_train_dataset.resize(x_train_dataset.shape[0] + all_train_data.shape[0], axis=0)
                    y_train_dataset.resize(y_train_dataset.shape[0] + all_train_target.shape[0], axis=0)
                    x_test_dataset.resize(x_test_dataset.shape[0] + all_test_data.shape[0], axis=0)
                    y_test_dataset.resize(y_test_dataset.shape[0] + all_test_target.shape[0], axis=0)
                    
                    # Append new patches to x_train, y_train, x_test, and y_test
                    x_train_dataset[-all_train_data.shape[0]:] = all_train_data.cpu().numpy()
                    y_train_dataset[-all_train_target.shape[0]:] = all_train_target.cpu().numpy()
                    x_test_dataset[-all_test_data.shape[0]:] = all_test_data.cpu().numpy()
                    y_test_dataset[-all_test_target.shape[0]:] = all_test_target.cpu().numpy()
                    
                    allocated_memory = torch.cuda.memory_allocated(device)
                    reserved_memory = torch.cuda.memory_reserved(device)
                    reserved_fraction = reserved_memory / total_memory
                    if reserved_fraction > memory_threshold:
                        print(f"Memory usage is high ({reserved_fraction*100:.2f}%). Clearing cache...")
                        torch.cuda.empty_cache()
                    else:
                        print(f"Memory usage is at {reserved_fraction*100:.2f}%. No need to clear cache.")
    
    return h5_file_path