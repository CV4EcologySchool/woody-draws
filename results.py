import yaml
from train import create_dataloader, load_model

# load config
config = "configs/data_config.yaml"
print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))


# setup entities
dl_test = create_dataloader(cfg, split='train')

# load model
model = load_model(cfg)
train_loss = []
val_loss = []
for i in range(0, cfg["num_epochs"]):
    epoch = torch.load("model_states/{}.pt".format(i+1))
    train_loss.append(epoch["loss_train"])
    val_loss.append(epoch["loss_val"])

epochs = [i for i in range(0, cfg["num_epochs"])]
plt.plot(epochs, train_loss)
plt.savefig('figs/losses.png')


from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

y_pred = []
y_true = []

# iterate over test data
for i, (inputs, labels) in enumerate(dl_test):
        print("Working on batch {} of {}".format(i, len(dl_test)))
        output = model[0](inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

classes = dl_test.dataset.le.classes_
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('figs/confusion_matrix.png')





