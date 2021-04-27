import torch
import time
from network import NN
from ipdb import set_trace

#-----------------------
# 建立假數據
#    負樣本(x0, y0)
#    正樣本(x1, y1)
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100, 1)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100, 1)

# 合併正負樣本
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.FloatTensor)


#---------------------
# Training
model = NN(2, 10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_func = torch.nn.BCEWithLogitsLoss()
s = time.time()
for t in range(10000):
    out = model(x)

    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t%1000==0:
        print('[iter-{}] loss = {}'.format(t, loss.item())) 
e = time.time()
print('time =', e-s)
