import torch
import time
from torch.utils.data import DataLoader
from network import NN
from mydataset import MyDataset
from ipdb import set_trace

#-----------------------
# 建立假數據
#    負樣本(x0, y0)
#    正樣本(x1, y1)
n_data = torch.ones(100000, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(10000, 1)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(10000, 1)

# 合併正負樣本
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.FloatTensor)

device = 'cuda:0'

mydataset = MyDataset(x, y)
dataloader = DataLoader(
    dataset=mydataset,
    batch_size=512,
    shuffle=True,
    num_workers=2,
)
#---------------------
# Training
model = NN(2, 10, 1)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
loss_func = torch.nn.BCEWithLogitsLoss()
s = time.time()

for epoch in range(100):
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)

        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('[epoch: {}] loss = {}'.format(epoch, loss.item())) 
e = time.time()
print('time =', e-s)
