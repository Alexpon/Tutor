import os
import torch
import pandas as pd

from torch.utils.data import DataLoader
from network import CNN
from mydataset import MyDataset, MyValDataset

from ipdb import set_trace


sigmoid_fn = torch.nn.Sigmoid()

def validate(dataloader, model, device):
    model.eval()
    num_data    = 0
    num_correct = 0
    for data, label in val_dataloader:
        data  = data.to(device)
        label = label.to(device)
        
        output = model(data)
        output = output.detach()
        output = output.squeeze()
        pred_prob = sigmoid_fn(output)
        pred_lab  = (pred_prob>0.5).int()
        num_data    += label.shape[0]
        num_correct += (pred_lab==label).sum().item()
    val_accuracy = num_correct/num_data
    print ('[val] Accuracy =', val_accuracy)
    return val_accuracy

def test(test_dataloader, model, device):
    model.eval()
    num_data    = 0
    num_correct = 0
    for data, label in test_dataloader:
        data  = data.to(device)
        label = label.to(device)
        
        output = model(data)
        output = output.detach()
        output = output.squeeze()
        pred_prob = sigmoid_fn(output)
        pred_lab  = (pred_prob>0.5).int()
        num_data    += label.shape[0]
        num_correct += (pred_lab==label).sum().item()
    test_accuracy = num_correct/num_data
    print ('[test] Accuracy =', test_accuracy)

# 讀csv檔
df = pd.read_csv('../data/dog_wolf_small/data.csv')
filename = df.img_path.tolist()
labels   = df.label.tolist()

mydataset = MyDataset(filename, labels)
dataloader = DataLoader(
    dataset=mydataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
)


df = pd.read_csv('../data/dog_wolf_small_v2/data.csv')
val_filename = df.img_path.tolist()
val_labels   = df.label.tolist()
my_val_dataset = MyValDataset(val_filename, val_labels)
val_dataloader = DataLoader(
    dataset=my_val_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

#---------------------
# Training
model = CNN()
model = model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):
    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = loss_func(out, label.float().unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch-{} loss = {}'.format(epoch, loss.item()))
    
    if epoch%2==0:
        val_accuracy = validate(val_dataloader, model, device)

    if epoch%2==0:
        state = {
            'epoch'                :  epoch,
            'policy_state_dict'    :  model.state_dict(),
            'optimizer_state_dict' :  optimizer.state_dict(),
            'val_accuracy'         :  val_accuracy,
        }
        model_dir = './model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        ckpt_path = os.path.join(model_dir, 'epoch-{}.ckpt'.format(epoch))
        torch.save(state, ckpt_path)
    
test(dataloader, model, device)
