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

def replace_other(row):
    if row["dom_overstory"] == "other":
        return row["overstory_other"]

vf = "data/vectors/draw_polygons.geojson"
field_data_file = "data/vectors/WD_Field_Data_20220803.geojson"
image_glob = "data/images/*oid*.tif"

draws = gpd.read_file(vf)
transects = gpd.read_file(field_data_file)#[["draw", "dom_overstory", "overstory_other"]]
mulligans = transects["notes"].str.contains("Mulligan")
transects = transects[np.logical_not(mulligans)]
transects = transects[np.logical_not(transects["point_1_other"].str.contains("/a"))]
transects = transects[np.logical_not(transects["dom_overstory"].str.contains("/A"))]
transects = transects[~transects["draw"].isna()]
transects["dom_overstory"] = np.where(transects["dom_overstory"] == "other", transects["overstory_other"], transects["dom_overstory"])
transects["dom_overstory"] = transects["dom_overstory"].replace('Pinus ponderosa var. scopulorum', "Pinus ponderosa")

transects = transects[["draw", "dom_overstory"]]
transects = transects[["draw", "dom_overstory"]].dropna()

transects["draw"] = transects["draw"].astype(int)
image_files = glob(image_glob)

object_ids = pd.DataFrame([(f, int(f.split("oid")[1][:-4]), int(f.split("_")[-2])) for (i,f) in enumerate(image_files)])
object_ids.columns = ["file_path", "oid", "cid"]
sampling_table = pd.merge(draws, object_ids, how = "right", left_on = "OBJECTID", right_on = "oid")[["file_path", "oid", "cid", "k"]]
sampling_table = pd.merge(sampling_table, transects, how = "left", left_on = "oid", right_on = "draw")

data = sampling_table[sampling_table["dom_overstory"].notnull()]










