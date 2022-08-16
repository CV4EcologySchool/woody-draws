import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from scipy.special import softmax
from PIL import Image
from draw_classifier.make_partition_table import make_partition_table
import yaml
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
plt.rcParams['figure.figsize'] = [15, 7]
plt.rcParams['figure.dpi'] = 100
#c = "configs/curriculum_test.yaml"
#cfg = yaml.safe_load(open(c, "r"))

class WDDataSet(Dataset):
    def __init__(self, cfg: dict, split: str = "train", epoch_number: int = 1):
        self.split = split
        if split == "train":
            self.transform = A.Compose([
                A.ToFloat(max_value = 255.0),
                A.HorizontalFlip(p=cfg["horizontal_flip_p"]),
                A.VerticalFlip(p=cfg["vertical_flip_p"]),
                ToTensorV2()])
        else:
            self.transform = A.Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
                A.ToFloat(max_value = 255.0),
                ToTensorV2(),
            ])        
        self.draw_polygons = cfg["draw_polygons"]
        self.transects = cfg["transects"]
        self.image_glob = cfg["image_glob"]
        self.pred_col = cfg["pred_col"] 
        
        ### Check to see if we are using all the classes, or just the top-n most frequent classes
        try:
            self.global_table = make_partition_table(self.draw_polygons, self.transects, self.image_glob, self.pred_col)
            self.target_species = self.global_table[self.pred_col].value_counts()[:cfg["keep_top"]].index.tolist()
            self.global_table = self.global_table[self.global_table[self.pred_col].isin(self.target_species)]
        except KeyError:
            self.global_table = make_partition_table(self.draw_polygons, self.transects, self.image_glob, self.pred_col)
        
        
        self.data_table = self.global_table[self.global_table["split"] == self.split]
        self.ordered_classes = list(self.data_table[self.pred_col].value_counts().index)
        if cfg["curriculum"] == True and split == "train":
            self.true_class_probabilities = np.array(self.data_table[self.pred_col].value_counts() / self.data_table.shape[0])
            self.ordered_classes_sample_probs = softmax(((1/self.true_class_probabilities)/epoch_number) + self.true_class_probabilities)
            self.data_table["weight"] = np.nan 
            for i,(c,prob) in enumerate(zip(self.ordered_classes, self.ordered_classes_sample_probs)):
                self.data_table.loc[(self.data_table[self.pred_col] == c), "weight"] = prob
                                        
            self.data_table = self.data_table.sample(n = int(self.data_table.shape[0]), weights = self.data_table["weight"], random_state = 4818)

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
        ds_arr = ds.ReadAsArray()
        ds_arr = ds_arr.astype(np.uint8).reshape(self.img_width, self.img_height, self.n_channels)
        img_tensor = self.transform(force_apply = False, image=ds_arr)
        img_tensor = img_tensor["image"]
        
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
    train = WDDataSet(cfg, split = "train", epoch_number = 5)
    print(train.data_table)
    #train.plot_split(x = cfg["pred_col"], to_file = "eval/{}/train_strata_split.png".format(cfg["experiment_name"]))
    






