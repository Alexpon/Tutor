import torch
import pandas as pd

from torch.utils.data import DataLoader
from network import CNN
from mydataset import MyDataset

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
        
    print ('[val] Accuracy =', num_correct/num_data)


# 讀csv檔
df = pd.read_csv('./data/dog_wolf_small/data.csv')
filename = df.img_path.tolist()
labels   = df.label.tolist()

mydataset = MyDataset(filename, labels)
dataloader = DataLoader(
    dataset=mydataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
)

val_dataloader = DataLoader(
    dataset=mydataset,
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

for epoch in range(1000):
    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = loss_func(out, label.float().unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch%2==0:
        validate(val_dataloader, model, device)
    print('epoch-{} loss = {}'.format(epoch, loss.item()))

