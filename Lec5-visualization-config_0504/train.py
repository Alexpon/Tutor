import os
import yaml
import torch
import pandas as pd

from torch.utils.data import DataLoader
from network import CNN
from mydataset import MyDataset
from plotter import Plotter
from ipdb import set_trace


sigmoid_fn = torch.nn.Sigmoid()
loss_func = torch.nn.BCEWithLogitsLoss()

def validate(dataloader, model, device):
    model.eval()
    num_data    = 0
    num_correct = 0
    loss_sum    = 0
    for data, label in val_dataloader:
        data  = data.to(device)
        label = label.to(device)

        batch_size = label.shape[0]
        output    = model(data)
        loss      = loss_func(out, label.float().unsqueeze(1)).item()
        loss_sum += loss*batch_size

        output    = output.detach()
        output    = output.squeeze()
        pred_prob = sigmoid_fn(output)
        pred_lab  = (pred_prob>0.5).int()
        
        num_data    += batch_size
        num_correct += (pred_lab==label).sum().item()
    
    val_accuracy = num_correct/num_data
    val_loss     = loss_sum/num_data
    return val_accuracy, val_loss

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



config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)

# 讀csv檔
df = pd.read_csv(config['data']['filepath'])
filename = df.img_path.tolist()
labels   = df.label.tolist()

mydataset = MyDataset(filename, labels)
dataloader = DataLoader(
    dataset     = mydataset,
    batch_size  = config['train']['batch'],
    shuffle     = True,
    num_workers = 2,
)

val_dataloader = DataLoader(
    dataset=mydataset,
    batch_size=config['train']['batch'],
    shuffle=True,
    num_workers=2,
)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

plotter = Plotter(exp_name = 'env_0504')

#---------------------
# Training
model = CNN()
model = model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(config['train']['epoch']):
    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = loss_func(out, label.float().unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('epoch-{}'.format(epoch))
    print('\t[train] loss =',loss.item())
    if epoch%2==0:
        val_accuracy, val_loss = validate(val_dataloader, model, device)
        plotter.plot(val_accuracy, val_loss, epoch, phase='val')
        print('\t[val] accuracy =', val_accuracy)
        print('\t[val]     loss =', val_loss)

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

