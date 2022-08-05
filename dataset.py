import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from make_partition_table import make_partition_table
import yaml
import numpy as np
from osgeo import gdal

cfg = yaml.safe_load(open("configs/data_config.yaml", "r"))

class WDDataSet(Dataset):

    def __init__(self, cfg, split = "train"):
        self.split = split
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])        
        self.draw_polygons = cfg["draw_polygons"]
        self.transects = cfg["transects"]
        self.image_glob = cfg["image_glob"]
        
        
        self.global_table = make_partition_table(self.draw_polygons, self.transects, self.image_glob)
        self.data_table = self.global_table[self.global_table["split"] == self.split]
        self.images = self.data_table["file_path"]
        self.labels = self.data_table["dom_overstory"]
        self.data = [(i,l) for i, l in zip(self.images, self.labels)]

    def __len__(self):
        return self.data_table.shape[0]

    def __getitem__(self, idx):
        image_name, label = self.data[idx]
        ds = gdal.Open(image_name)
        ds_arr = ds.ReadAsArray().astype(np.uint8)
        img_tensor = self.transform(ds_arr)
        return img_tensor, label

if __name__ == "__main__":
    dataset = WDDataSet(cfg)
    print(dataset[32])
