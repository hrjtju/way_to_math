import os
import json
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import yaml
from utils import FilterSliceBuilder, SliceBuilder, get_slice_builder, mirror_pad

class RawMSDLungPatchDataset(Dataset):
    def __init__(self, base_path, json_path, train=True, transform=None, slice_builder_config=None):
        # define augmentation transforms
        self.transform = transform
        self.train = train
        self.slice_builder_config = slice_builder_config or {}
        
        # load json file containing dataset information
        with open(json_path, 'r') as json_file:
            self.json_info = json.load(json_file)
        self.base_path = base_path
        
        # train / test set file list
        self.data_file_list = self.json_info["training"] if self.train else self.json_info["test"]
        
        # constructing patch index
        self.patch_index = []  # [(volume_idx, slice_obj), ...]
        self._build_patch_index()
        
        print(f"Total patches: {len(self.patch_index)}")
        
    def _build_patch_index(self):
        """预计算所有体积中的所有patch索引"""
        for volume_idx, file_info in enumerate(self.data_file_list):
            img_path = os.path.join(self.base_path, file_info['image'])
            
            # 只加载shape信息，不加载完整数据
            nib_img = nib.load(img_path)
            volume_shape = nib_img.dataobj.shape  # (Z, Y, X)
            
            # 创建伪HDF5接口（适配SliceBuilder）
            class PseudoH5Dataset:
                def __init__(self, shape, ndim):
                    self.shape = shape
                    self.ndim = ndim
            
            # 根据数据维度调整（单通道CT）
            ndim = 3  # MSD Lung是3D体积
            raw_h5 = PseudoH5Dataset(volume_shape, ndim=ndim)
            label_h5 = None if not self.train else PseudoH5Dataset(volume_shape, ndim=ndim)
            
            # 构建该体积的SliceBuilder
            slice_builder = SliceBuilder(raw_h5, label_h5, **self.slice_builder_config)
            
            # 将patch索引添加到全局列表
            for slice_obj in slice_builder.raw_slices:
                self.patch_index.append((volume_idx, slice_obj))
    
    def __len__(self):
        return len(self.patch_index)
    
    def _load_volume(self, volume_idx):
        """按需加载单个体积"""
        img_path = os.path.join(self.base_path, self.data_file_list[volume_idx]['image'])
        img = nib.load(img_path).get_fdata().astype(np.float32)
        
        if self.train:
            label_path = os.path.join(self.base_path, self.data_file_list[volume_idx]['label'])
            label = nib.load(label_path).get_fdata().astype(np.int64)
            return img, label
        return img, None
    
    def __getitem__(self, idx):
        # 获取patch对应的体积索引和切片对象
        volume_idx, slice_obj = self.patch_index[idx]
        
        # 加载该体积（带缓存机制可优化）
        img, label = self._load_volume(volume_idx)
        
        # 提取patch
        img_patch = img[slice_obj]
        
        # 测试模式：添加halo并返回原始索引用于拼接
        if not self.train and self.slice_builder_config.get('halo_shape', [0,0,0]) != [0,0,0]:
            halo_shape = self.slice_builder_config['halo_shape']
            img_patch = mirror_pad(img_patch, halo_shape)
            slice_obj_padded = tuple(slice(s.start, s.stop + 2*h) for s, h in zip(slice_obj, halo_shape))
            return self.transform["raw"](img_patch), slice_obj
        
        # 训练模式：提取标签patch
        if self.train:
            label_patch = label[slice_obj]
            
            # 应用变换
            if self.transform:
                img_patch = self.transform["raw"](img_patch)
                label_patch = self.transform["label"](label_patch)
            
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