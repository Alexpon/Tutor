import os
import yaml
import torch
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models
from mydataset import MyDataset, MyValDataset
from plotter import Plotter



sigmoid_fn = torch.nn.Sigmoid()
loss_func = torch.nn.BCEWithLogitsLoss()
module_out = []

# signatures are fixed
def hook_fn(module, input, output):
    output = output.squeeze(3).squeeze(2)
    module_out.append(output)


def save_data():
    data = torch.cat(module_out).detach()
    data = data.numpy()
    df = pd.DataFrame(data)
    df.to_csv('hidden_output.csv')



def validate(dataloader, model, device):
    model.eval()
    # initialize module_out
    module_out = []
    num_data    = 0
    num_correct = 0
    loss_sum    = 0
    for data, label in dataloader:
        data  = data.to(device)
        label = label.to(device)
        batch = label.shape[0]

        output = model(data)
        loss   = loss_func(output, label.float().unsqueeze(1)).item()
        loss_sum += loss*batch
        output = output.detach()
        output = output.squeeze()
        pred_prob = sigmoid_fn(output)
        pred_lab  = (pred_prob>0.5).int()
        num_data    += batch
        num_correct += (pred_lab==label).sum().item()
    val_accuracy = num_correct / num_data
    val_loss     = loss_sum    / num_data
    print ('[val] Accuracy =', val_accuracy)
    print ('[val]     Loss =', val_loss)
    return val_accuracy, loss


def test(test_dataloader, model, device):
    model.eval()
    # initialize module_out
    module_out = []
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
    save_data()



def main():
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)

# 讀csv檔
    df = pd.read_csv(config['data']['train_path'])
    filename = df.img_path.tolist()
    labels   = df.label.tolist()

    mydataset = MyDataset(filename, labels, transform_type=config['train']['transform_type'])
    dataloader = DataLoader(
        dataset     = mydataset,
        batch_size  = config['train']['batch'],
        shuffle     = True,
        num_workers = 2,
    )


    df = pd.read_csv(config['data']['val_path'])
    val_filename = df.img_path.tolist()
    val_labels   = df.label.tolist()
    val_dataset = MyValDataset(val_filename, val_labels)
    val_dataloader = DataLoader(
        dataset     = val_dataset,
        batch_size  = config['train']['batch'],
        shuffle     = True,
        num_workers = 2,
    )

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    if config['plot']:
        plotter = Plotter('dog_wolf_with_aug')

#---------------------
# Training
    #model = CNN()
    model  = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 1)
    
    model.avgpool.register_forward_hook(hook_fn)

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config['train']['epoch']):
        # initialize module_out
        module_out = []
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = loss_func(out, label.float().unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch-{} train loss = {}'.format(epoch, loss.item()))
        
        val_accuracy, val_loss = validate(val_dataloader, model, device)
        if config['plot']:
            plotter.plot(val_accuracy, val_loss, epoch, phase='val')

        if config['save_model']:
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

if __name__=='__main__':
    main()
