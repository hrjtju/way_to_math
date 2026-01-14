from collections import OrderedDict
import os
import h5py
from torch.utils.data import Dataset
import nibabel as nib
import json
import numpy as np
from tqdm import tqdm
import yaml
from functools import lru_cache
import pickle

from utils import FilterSliceBuilder, SliceBuilder, get_slice_builder, mirror_pad

class RawMSDLungDataset(Dataset):
    def __init__(self, base_path, json_path, train=True, transform=None):
        # define augmentation transforms
        self.transform = transform
        
        self.data_list = []
        self.train = train
        
        # load json file containing dataset information
        with open(json_path, 'r') as json_file:
            self.json_info = json.load(json_file)
        self.base_path = base_path
        
        # length of the dataset
        self.len = self.json_info["numTraining"] if self.train else self.json_info["numTest"]
        self.data_file_list = self.json_info["training"] if self.train else self.json_info["test"]

        
    def __len__(self):
        return self.len

    def add_base(self, relative_path):
        return os.path.join(self.base_path, relative_path)
    
    def __getitem__(self, idx):
        # fetch image and label paths
        img_path = self.add_base(self.data_file_list[idx]['image'])
        img = nib.load(img_path).get_fdata()
        
        if self.transform:
            img = self.transform["raw"](img)
        
        if self.train:
            label_path = self.add_base(self.data_file_list[idx]['label'])
            label = nib.load(label_path).get_fdata()
            
            if self.transform:
                label = self.transform["label"](label)
            
            return img, label
        else:
            # skip the label in testing mode
            return img
class RawMSDLungPatchDataset(Dataset):
    def __init__(self, base_path, json_path, mode=True, transform=None, slice_builder_config=None):
        # define augmentation transforms
        self.transform = transform
        self.mode = mode
        self.slice_builder_config = slice_builder_config or {}
        
        # load json file containing dataset information
        with open(json_path, 'r') as json_file:
            self.json_info = json.load(json_file)
        self.base_path = base_path
        self.patch_index_path = os.path.join(self.base_path, f"patch_index_{self.mode}.pkl")
        # train / test set file list
        self.data_file_list = self.json_info[self.mode]
        
        # constructing patch index
        self.patch_index = []  # [(volume_idx, slice_obj), ...]
        
        try:
            print(f"Try loading patch index from {self.patch_index_path}")
            with open(self.patch_index_path, "rb") as f:
                self.patch_index = pickle.load(f)    
        except Exception as e:
            print(f"Loding failed due to >>> \n{e}\n...")
            print("Building patch index")
            self._build_patch_index()
        
        print(f"Total patches: {len(self.patch_index)}")
        
        self.cache_max_size = 16
        
        print("Setting LRU Cache")
        
        self._volume_cache: "OrderedDict[int, tuple[np.ndarray, np.ndarray|None]]" = OrderedDict()
        self._volume_cache_max = int(self.slice_builder_config.get("cache_size", self.cache_max_size))
        self._load_volume = self._load_volume_wrapper

    # use wrapper that is pickleable (no functools._lru_cache_wrapper bound on instance)
    # DataLoader workers can now pickle the dataset object on Windows.
    # Note: cached values are kept as numpy arrays; monitor memory if cache_size is large.
    def hashable(hashable, slice_obj):
        return tuple((s.start, s.step, s.stop) for s in slice_obj)
    
    def _load_volume_wrapper(self, volume_idx: int, slice_obj):
        # cache hit
        hash_slice = self.hashable(slice_obj)
        if ((volume_idx, hash_slice)) in self._volume_cache:
            img_lbl = self._volume_cache.pop((volume_idx, hash_slice))
            # move to end = most recently used
            self._volume_cache[(volume_idx, hash_slice)] = img_lbl
            return img_lbl

        # cache miss -> load and insert
        img_lbl = self._load_volume_impl(volume_idx, slice_obj)
        self._volume_cache[(volume_idx, hash_slice)] = img_lbl
        # enforce max size
        while len(self._volume_cache) > self._volume_cache_max:
            self._volume_cache.popitem(last=False)
        return img_lbl

    
    def _build_patch_index(self):
        """预计算所有体积中的所有patch索引"""
        
        for volume_idx, file_info in tqdm(enumerate(self.data_file_list), total=len(self.data_file_list)):
            if self.mode == "test":
                img_path = os.path.join(self.base_path, file_info)
            else:
                img_path = os.path.join(self.base_path, file_info['image'])
            
            # open the h5 files and pass real h5py.Dataset objects to the SliceBuilder
            # FilterSliceBuilder and friends index the dataset to filter patches, so
            # we must provide real dataset objects (not a pseudo wrapper).
            if self.mode == "test":
                f_img = h5py.File(img_path, 'r')
                raw_h5 = f_img['data']
                label_h5 = None
                # choose slice builder class: if config contains 'name', use get_slice_builder
                slice_builder = SliceBuilder(raw_h5, label_h5, **self.slice_builder_config)

                for slice_obj in slice_builder.raw_slices:
                    self.patch_index.append((volume_idx, slice_obj))

                # close the file after slices are materialized
                f_img.close()
            else:
                img_path = os.path.join(self.base_path, file_info['image'])
                label_path = os.path.join(self.base_path, file_info['label'])
                f_img = h5py.File(img_path, 'r')
                f_lbl = h5py.File(label_path, 'r')
                raw_h5 = f_img['data'][:].astype(np.float32).transpose(2, 1, 0)
                label_h5 = f_lbl['data'][:].astype(np.int64).transpose(2, 1, 0)

                slice_builder = FilterSliceBuilder(raw_h5, label_h5, **self.slice_builder_config)

                for slice_obj in slice_builder.raw_slices:
                    self.patch_index.append((volume_idx, slice_obj))

                # close both files after the slice list is built
                f_img.close()
                f_lbl.close()
            
        print(f"Saving seralized binary slice index to {self.patch_index_path}")
        
        with open(self.patch_index_path, 'wb') as f:
            pickle.dump(self.patch_index, f)
    
    def __len__(self):
        return len(self.patch_index)
    
    def _load_volume_impl(self, volume_idx, slice_obj):
        """按需加载单个体积"""
        file_info = self.data_file_list[volume_idx]
        
        h5_slice = (slice_obj[2], slice_obj[1], slice_obj[0])
        
        if self.mode == "test":
            img_path = os.path.join(self.base_path, file_info)
        else:
            img_path = os.path.join(self.base_path, file_info['image'])
            
        with h5py.File(img_path, 'r') as f:
            img = f['data'][h5_slice].astype(np.float32).transpose(2, 1, 0)
        
        if self.mode != "test":
            label_path = os.path.join(self.base_path, file_info['label'])
            with h5py.File(label_path, 'r') as f:
                label = f['data'][h5_slice].astype(np.int64).transpose(2, 1, 0)
            return img, label
        
        return img, None
    
    def __getitem__(self, idx):
        # 获取patch对应的体积索引和切片对象
        volume_idx, slice_obj = self.patch_index[idx]
        
        # 加载该体积（带缓存机制可优化）
        img_patch, label_patch = self._load_volume(volume_idx, slice_obj)
        
        # 测试模式：添加halo并返回原始索引用于拼接
        if self.mode == "test" and self.slice_builder_config.get('halo_shape', [0,0,0]) != [0,0,0]:
            halo_shape = self.slice_builder_config['halo_shape']
            img_patch = mirror_pad(img_patch, halo_shape)
            slice_obj_padded = tuple(slice(s.start, s.stop + 2*h) for s, h in zip(slice_obj, halo_shape))
            return self.transform["raw"](img_patch), slice_obj
        
        # 训练模式：提取标签patch
        if self.mode != "test":
            
            # 应用变换
            if self.transform:
                img_patch = self.transform["raw"](img_patch)
                label_patch = self.transform["label"](label_patch)
            
            # print(img_patch.max(), img_patch.min(), label_patch.max(), label_patch.min())
            
            return img_patch, label_patch
        
        # 测试模式（无halo）
        if self.transform:
            img_patch = self.transform["raw"](img_patch)
        
        return img_patch

if __name__ == "__main__":
    base_path = r"E:\2_Project_Data\3_SYSU_MLLecture_final_medical_segmentation_lung_ca\0_original\Task06_Lung"
    json_path = os.path.join(base_path, 'dataset.json')

    with open(r"D:\Research\Mathematics\SYSU_GBU\6_Lectures\0_MachineLearning\2_Final_homework\2_code\3d_unet_cfg.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    dataset = RawMSDLungPatchDataset(base_path=base_path, json_path=json_path, train=True,
                                     slice_builder_config=cfg["loaders"]["train"]["slice_builder"])
    print(f"Dataset length: {len(dataset)}")
    sample_img, sample_label = dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample label shape: {sample_label.shape}")
