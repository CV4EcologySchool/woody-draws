"""
Makes a table with object ID, stratum, label, and partition,
to be used with custom data loader as described in here:
    https://stackoverflow.com/questions/65231299/load-csv-and-image-dataset-in-pytorch

"""

import geopandas as gpd
import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import argparse
import yaml
#### to do
### make input/output from config file on command line

cfg = yaml.safe_load(open("configs/augarg_test.yaml", "r"))
image_glob = cfg["image_glob"]
transectsf = cfg["transects"]
drawsf = cfg["draw_polygons"]
pred_col = cfg["pred_col"]
def make_partition_table(drawsf, transectsf, image_glob, pred_col):
    draws = gpd.read_file(drawsf)
    transects = gpd.read_file(transectsf)#[["draw", "dom_overstory", "overstory_other"]]
    mulligans = transects["notes"].str.contains("Mulligan")
    transects = transects[np.logical_not(mulligans)]
    transects = transects[np.logical_not(transects["point_1_other"].str.contains("/a"))]
    transects = transects[np.logical_not(transects["dom_overstory"].str.contains("/A"))]
    transects = transects[~transects["draw"].isna()]
    transects["dom_overstory"] = np.where(transects["dom_overstory"] == "other", transects["overstory_other"], transects["dom_overstory"])
    transects["dom_overstory"] = transects["dom_overstory"].replace('Pinus ponderosa var. scopulorum', "Pinus ponderosa")

    transects = transects[["draw", pred_col]]
    transects = transects[["draw", pred_col]].dropna()

    transects["draw"] = transects["draw"].astype(int)
    image_files = glob(image_glob)

    object_ids = pd.DataFrame([(f, int(f.split("oid")[1][:-4]), int(f.split("_")[-2]), int(f.split("_")[5][0:4])) for (i,f) in enumerate(image_files)])
    object_ids.columns = ["file_path", "oid", "cid", "date"]
    sampling_table = pd.merge(draws, object_ids, how = "right", left_on = "OBJECTID", right_on = "oid")[["file_path", "oid", "cid", "k", "date"]]
    sampling_table = pd.merge(sampling_table, transects, how = "left", left_on = "oid", right_on = "draw")

    data = sampling_table[sampling_table[pred_col].notnull()]


    for_splitting = data[["oid", "k", "date"]].drop_duplicates()
    n_draws = for_splitting.shape[0]
    for_splitting['group'] = for_splitting.groupby(['k', 'date'], sort=False).ngroup() + 1
    for_splitting = for_splitting[for_splitting["group"] != 1]
    for_splitting = for_splitting[for_splitting["group"] != 6]
    for_splitting = for_splitting[for_splitting["group"] != 7]
    train, test = train_test_split(for_splitting, test_size=0.6, random_state=47, stratify=for_splitting[['group']])
    train["split"] = "train"
    #test = test[test["group"] != 5]
    
    test, val = train_test_split(test, test_size = 0.5, random_state=331, stratify=test["group"])
    test["split"] = "test"
    val["split"] = "val"
    for_splitting = pd.concat([train,test,val], axis = 0)
    final_dataset = pd.merge(for_splitting, data, how = "right").dropna()
    return final_dataset


