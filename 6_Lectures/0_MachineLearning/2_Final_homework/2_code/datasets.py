import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy import ndimage
import json

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

if __name__ == "__main__":
    base_path = r"E:\2_Project_Data\3_SYSU_MLLecture_final_medical_segmentation_lung_ca\0_original\Task06_Lung"
    json_path = os.path.join(base_path, 'dataset.json')

    dataset = RawMSDLungDataset(base_path=base_path, json_path=json_path, train=True)
    print(f"Dataset length: {len(dataset)}")
    sample_img, sample_label = dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample label shape: {sample_label.shape}")