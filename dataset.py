import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from make_partition_table import make_partition_table
import yaml
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder

plt.rcParams['figure.figsize'] = [15, 7]
plt.rcParams['figure.dpi'] = 100

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
        self.le = LabelEncoder().fit(self.labels)
        self.labels = self.le.transform(self.labels)
        self.data = [(i,l) for i, l in zip(self.images, self.labels)]

        self.img_width = cfg["img_width"]
        self.img_height = cfg["img_height"]
        self.n_channels = cfg["n_channels"]

    def __len__(self):
        return self.data_table.shape[0]

    def __getitem__(self, idx):
        image_name, label = self.data[idx]
        ds = gdal.Open(image_name)
        ds_arr = ds.ReadAsArray().astype(np.uint8).reshape(self.img_width, self.img_height, self.n_channels)
        img_tensor = self.transform(ds_arr)
        return img_tensor, label

    def plot_split(self, x = "k", hue = "dom_overstory", to_file = ""):
        plt.clf()
        sns.countplot(data=self.data_table, x=x, hue= hue)
        plt.gca().set_xlabel("Sampling Stratum")        
        if len(to_file) > 1:
            plt.savefig(to_file)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/data_config.yaml", "r"))
    dataset = WDDataSet(cfg, split = "train")
    dataset.plot_split(to_file = "figs/train_strata_split.png")
    dataset = WDDataSet(cfg, split = "test")
    dataset.plot_split(to_file = "figs/test_strata_split.png")
