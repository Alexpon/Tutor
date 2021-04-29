import torch
import pandas as pd

from torch.utils.data import DataLoader
from network import CNN
from mydataset import MyDataset


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

device = 'cuda:0'

#---------------------
# Training
model = CNN()
model = model.to(device)
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
    print('epoch-{} loss = {}'.format(epoch, loss.item()))

