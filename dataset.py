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
        self.pred_col = cfg["pred_col"] 
        
        self.global_table = make_partition_table(self.draw_polygons, self.transects, self.image_glob, self.pred_col)
        self.data_table = self.global_table[self.global_table["split"] == self.split]
        self.images = self.data_table["file_path"]
        self.labels = self.data_table[self.pred_col]
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

    def plot_split(self, x = "", to_file = ""):
        plt.clf()
        #sns.countplot(data=self.data_table, x=x, hue= hue) 
        fig, ax = plt.subplots(1,3, figsize=(14,8))
        unique = self.global_table[x].unique()
        palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))

        y_train = self.global_table[self.global_table.split == "train"]
        y_test = self.global_table[self.global_table.split == "test"]
        y_val = self.global_table[self.global_table.split == "val"]
        for idx, group in enumerate([('Train', y_train), ('Test', y_test), ("Val", y_val)]):
            data = group[1][x].value_counts()
            sns.barplot(ax=ax[idx], x=data.index, y=data.values, palette=palette)
            ax[idx].set_title(f'{group[0]} Label Count')
            ax[idx].set_xlabel(f'{group[0]} Labels')
            ax[idx].set_ylabel('Label Count')
            ax[idx].tick_params(labelrotation=90)
        if len(to_file) > 1:
            plt.tight_layout()
            plt.savefig(to_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    train = WDDataSet(cfg, split = "train")
    train.plot_split(x = cfg["pred_col"], to_file = "figs/{}_train_strata_split.png".format(cfg["experiment_name"]))
    






