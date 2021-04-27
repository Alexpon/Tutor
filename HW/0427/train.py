import torch
import time
import pandas as pd

from torch.utils.data import DataLoader
from network import CNN
from mydataset import MyDataset
from ipdb import set_trace


# 讀csv檔
df = pd.read_csv('./data/dog_wolf_small/data.csv')
filename = df.img_path.tolist()
labels   = df.label.tolist()

mydataset = MyDataset(filename, labels)
dataloader = DataLoader(
    dataset=mydataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
)

device = 'cuda:0'

#---------------------
# Training
model = CNN()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_func = torch.nn.BCEWithLogitsLoss()
s = time.time()

for img, label in dataloader:
    img = img.to(device)
    label = label.to(device)
    out = model(img)
    loss = loss_func(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



