import yaml
from draw_classifier.train import create_dataloader, load_model
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import seaborn as sns
import pandas as pd
import numpy as np
import os


# setup entities
#c = "configs/actual_resnet50_curriculum_test.yaml"
#cfg = yaml.safe_load(open(c))
def plot_model_over_epochs(cfg, y_vars = ["loss", "oa"]):
    # load model
    model = load_model(cfg, find_best = False)
    train_loss = []
    val_loss = []
    train_oa = []
    val_oa = []

    for i in range(0, cfg["num_epochs"]):
        epoch = torch.load("model_states/{}/{}.pt".format(cfg["experiment_name"], i+1))
        train_loss.append(epoch["loss_train"])
        val_loss.append(epoch["loss_val"])
        train_oa.append(epoch["oa_train"])
        val_oa.append(epoch["oa_val"])

    epochs = [i for i in range(0, cfg["num_epochs"])]
    if "loss" in y_vars:
        plt.clf()
        plt.plot(epochs, train_loss, label = "train loss")
        plt.plot(epochs, val_loss, label = "validation loss")
        plt.legend()
        plt.savefig('eval/{}/losses.png'.format(cfg["experiment_name"]))
    if "oa" in y_vars:
        plt.clf()
        plt.plot(epochs, train_oa, label = "train accuracy")
        plt.plot(epochs, val_oa, label = "validation accuracy")
        plt.legend()
        plt.savefig('eval/{}/accuracies.png'.format(cfg["experiment_name"]))

# iterate over test data
def make_prediction_table(cfg, predict_proba = True, decode_labels = True):
    model = load_model(cfg, find_best = True)

    dl = create_dataloader(cfg, split='test')
    y_pred = []
    y_true = []
    y_predprobs = np.ndarray((0,int(cfg["num_classes"])), dtype=np.float32)
    for i, (inputs, labels) in enumerate(dl):
        print("Working on batch {} of {}".format(i, len(dl)))
        raw_output = model[0](inputs) # Feed Network
        
        output = (torch.max(torch.exp(raw_output), 1)[1]).data.cpu().numpy()
        if predict_proba:
            y_predprobs = np.vstack((y_predprobs, torch.softmax(raw_output, 1).data.cpu().numpy()))
        #output2 = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction            
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
    if decode_labels:
        y_pred = dl.dataset.le.inverse_transform(y_pred)
        y_true = dl.dataset.le.inverse_transform(y_true)
    if predict_proba:
        prediction_df = pd.DataFrame(y_predprobs, columns = dl.dataset.le.classes_)
        prediction_df["true"] = y_true
        prediction_df["predicted"] = y_pred
    else:
        prediction_df = pd.DataFrame({"true": y_true, "predicted": y_pred})
    return prediction_df

def make_classification_report(y_true,  y_pred, cfg):
    cr = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
    cr.to_csv("eval/{}/classification_report.csv".format(cfg["experiment_name"]))

def make_confusion_matrix(y_true, y_pred, cfg, **kwargs):
    dl = create_dataloader(cfg)
    cf_matrix = confusion_matrix(y_true, y_pred, labels = dl.dataset.le.classes_)
    plt.clf()
    plt.figure(figsize = (12,7))
    sns.heatmap(cf_matrix, annot=True)
    plt.gca().set_xlabel('Predicted labels')
    plt.gca().set_ylabel('True labels');
    #plt.gca().set_xticklabels(dl.dataset.le.classes_)
    #plt.gca().set_yticklabels(dl.dataset.le.classes_)
    plt.savefig('eval/{}/confusion_matrix.png'.format(cfg["experiment_name"]))

def make_auc_roc_curve(df, cfg):
    stopping_col = np.where(df.columns == "true")[0][0]
    pred_probs = df.iloc[:,0:stopping_col]    
    y_pred = df["predicted"]
    y_true = df["true"]
    plt.clf()
    false_positive = dict()
    true_positive = dict()
    for c in list(pred_probs.columns):
        this_class_y_true = np.where(y_true == c, 1, 0)
        false_positive[c], true_positive[c], threshs = roc_curve(this_class_y_true,
                                                            pred_probs[c])
        plt.plot(threshs[threshs <= 1.0], false_positive[c][threshs <= 1.0], lw=2, label='False positives: {}'.format(c))
        plt.plot(threshs[threshs <= 1.0], true_positive[c][threshs <= 1.0], lw=2, label='True positives: {}'.format(c))
        
    plt.xlabel("Threshold")
    plt.ylabel("True/False Positive Rate")
    plt.legend(loc="best")
    plt.title("ROC")
    plt.savefig("eval/{}/ROC_curve.png".format(cfg["experiment_name"]))

def make_precision_recall_curve(df, cfg):
    stopping_col = np.where(df.columns == "true")[0][0]
    pred_probs = df.iloc[:,0:stopping_col]    
    y_pred = df["predicted"]
    y_true = df["true"]
    plt.clf()
    precision = dict()
    recall = dict()
    for c in list(pred_probs.columns):
        this_class_y_true = np.where(y_true == c, 1, 0)
        precision[c], recall[c], _ = precision_recall_curve(this_class_y_true,
                                                            pred_probs[c])
        plt.plot(recall[c], precision[c], lw=2, label='class {}'.format(c))
        
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig("eval/{}/pr_curve.png".format(cfg["experiment_name"]))

def make_experiment_eval_folder(cfg):
    if not os.path.exists("eval/{}".format(cfg["experiment_name"])):
        os.makedirs("eval/{}".format(cfg["experiment_name"]))
    else:
        print("Found folder: eval/{}".format(cfg["experiment_name"]))

if __name__ == "__main__":
    # load config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = args.config
    print(f'Using config "{config}"')
    cfg = yaml.safe_load(open(config, 'r'))
    make_experiment_eval_folder(cfg)
    
    plot_model_over_epochs(cfg)
    predictions = make_prediction_table(cfg, decode_labels = True, predict_proba = True)
    make_auc_roc_curve(predictions, cfg)
    make_precision_recall_curve(predictions, cfg)
    make_confusion_matrix(predictions["true"], predictions["predicted"], cfg)
    make_classification_report(predictions["true"], predictions["predicted"], cfg)


