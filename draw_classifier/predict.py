import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from draw_classifier.make_partition_table import make_rf_partition_table
from draw_classifier.make_partition_table import make_partition_table
from draw_classifier.dataset import WDDataSet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from osgeo import gdal
import numpy as np
from glob import glob
from albumentations.pytorch import ToTensorV2
import albumentations as a
from torch.utils.data import DataLoader
import torch
from draw_classifier.model import define_resnet
from tqdm import trange
from scipy.special import softmax
import pandas as pd
import geopandas as gpd


class WDPredictionDataset(Dataset):
    def __init__(self, image_glob):
        self.images = glob(image_glob)
        self.transform = a.Compose([a.ToFloat(max_value = 255.0),
                                   ToTensorV2()])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        ds = gdal.Open(image_name)
        ds_arr = ds.ReadAsArray()
        ds_arr = ds_arr.astype(np.uint8).reshape(ds.RasterXSize, ds.RasterYSize, ds.RasterCount)
        img_tensor = self.transform(force_apply = False, image=ds_arr)
        img_tensor = img_tensor["image"]
        return img_tensor, image_name

def create_dataloader(cfg: dict):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = WDPredictionDataset(cfg["image_glob"])        # create an object instance of our CTDataset class
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers']
        )
    return dataLoader
def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    
    ### Instatiate model with specified backbone. For now, just keep everything at resnet50
    model_instance = define_resnet(cfg, "resnet50", pretrained=False)
    
    # load latest model state
    model_states = glob('model_states/{}/*.pt'.format(cfg["experiment_name"]))
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states/{}/'.format(cfg["experiment_name"]),'').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        experiment_name = cfg["experiment_name"]
        state = torch.load(open(f'model_states/{experiment_name}/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance


def run_predictions(cfg):
    cfg["curriculum"] = False
    device = cfg["device"]
    draw_polygons = cfg["draw_polygons"]
    transects = cfg["transects"]
    image_glob = cfg["image_glob"]
    pred_col = cfg["pred_col"]
    model = load_model(cfg)
    model.to(device)

    dataset = create_dataloader(cfg)
    all_predictions = []
    progressBar = trange(len(dataset))
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            predictions = list(model(batch[0]).cpu().numpy())
            #prediction = model(image.to(device))#.cpu().numpy()
            for pred, fname in zip(predictions, batch[1]):
                all_predictions.append([np.argmax(softmax(pred)), fname])
            progressBar.update(1)
    
    
    output_dataframe = pd.DataFrame(all_predictions)
    output_dataframe.columns = ["predicted", "filename"]
    output_dataframe["oid"] = output_dataframe["filename"].apply(lambda x: x.split("oid")[1].split(".")[0])
    dataset = WDDataSet(cfg)
    output_dataframe["predicted"] = dataset.le.inverse_transform(output_dataframe["predicted"])
    draws = gpd.read_file(draw_polygons)
    ground_truth = gpd.read_file(transects)
    ground_truth = ground_truth[["draw", cfg["pred_col"]]].dropna()
    ground_truth.columns = ["oid", "ground_truth"]
    output_dataframe = output_dataframe.groupby(["oid"])["predicted"].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    output_dataframe["oid"] = output_dataframe["oid"].astype(int)
    draws["OBJECTID"] = draws["OBJECTID"].astype(int)
    
    draws_with_preds = pd.merge(draws, output_dataframe, left_on = "OBJECTID", right_on = "oid", how = "right")    
    draws_with_preds = gpd.GeoDataFrame(pd.merge(ground_truth, draws_with_preds, how = "right")) 
    draws_with_preds.to_file("data/vectors/{}_with_model_predictions.geojson".format(cfg["experiment_name"]))    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    run_predictions(cfg)

if __name__=="__main__":
   main() 

