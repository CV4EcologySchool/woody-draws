import os
import argparse
import yaml
import glob
from tqdm import trange
from torchvision import models

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

# let's import our own classes and functions!
#from util import init_seed
from draw_classifier.dataset import WDDataSet
from draw_classifier.model import define_resnet
import numpy as np
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
torch.cuda.empty_cache()
#cfg = yaml.safe_load(open("configs/data_config.yaml", "r"))
#split = "train"
def create_dataloader(cfg: dict, epoch_number: int = 1, split: str = 'train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = WDDataSet(cfg, split, epoch_number)        # create an object instance of our CTDataset class
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader



def load_model(cfg, find_best = False):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    
    ### Instatiate model with specified backbone. For now, just keep everything at resnet50
    model_instance = define_resnet(cfg, "resnet50", pretrained=False)
    
    # load latest model state
    model_states = glob.glob('model_states/{}/[0-9]*.pt'.format(cfg["experiment_name"]))
    if len(model_states) > 0 and find_best == False:
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states/{}/'.format(cfg["experiment_name"]),'').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        experiment_name = cfg["experiment_name"]
        state = torch.load(open(f'model_states/{experiment_name}/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])
        return model_instance, start_epoch

    elif len(model_states) == 0 and find_best == False:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0
        return model_instance, start_epoch
    elif len(model_states) > 0 and find_best == True:
        model_epochs = [int(m.replace('model_states/{}/'.format(cfg["experiment_name"]),'').replace('.pt','')) for m in model_states]
        start_epoch = min(model_epochs)
        
        # load state dict and apply weights to model
        print(f'Searching for best model state  from epoch {start_epoch}')
        experiment_name = cfg["experiment_name"]
        epoch_idx = np.ones(len(model_epochs))*-1
        for i, epoch in enumerate(model_epochs):

            state = torch.load(open(f'model_states/{experiment_name}/{epoch}.pt', 'rb'), map_location='cpu')
            val_loss = state["loss_val"]
            print(val_loss)
            epoch_idx[i] = val_loss
        best_epoch = np.where(epoch_idx == epoch_idx.min())[0][0]
        state = torch.load(open(f'model_states/{experiment_name}/{best_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])
        torch.save(model_instance.state_dict(), open(f'model_states/{experiment_name}/best_model_epoch{best_epoch}.pt', 'wb'))
        return model_instance, best_epoch
        



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    experiment_name = cfg["experiment_name"]
    
    os.makedirs(f'model_states/{experiment_name}', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()
    # ...and save
    torch.save(stats, open(f'model_states/{experiment_name}/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = f'model_states/{experiment_name}/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

            

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer, current_epoch):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    if cfg["loss"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif cfg["loss"] == "kl_divergence":
        criterion = nn.KLDivLoss()

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
        del loss    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):
            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    np.random.seed(40182)

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for validation set
    dl_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')
        dl_train = create_dataloader(cfg, split='train', epoch_number = current_epoch+1)

        loss_train, oa_train = train(cfg, dl_train, model, optim, current_epoch+1)
        loss_val, oa_val = validate(cfg, dl_val, model)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
        save_model(cfg, current_epoch, model, stats)
    

    # That's all, folks!
        


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
