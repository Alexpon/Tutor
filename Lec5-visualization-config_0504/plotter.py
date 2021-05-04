from torch.utils.tensorboard import SummaryWriter

class Plotter():
    def __init__(self, exp_name):
        self.writer = SummaryWriter(log_dir='runs/{}'.format(exp_name))
    
    def plot(self, accuracy, loss, epoch, phase):
        self.writer.add_scalar('Accuracy/{}'.format(phase), accuracy, epoch)
        self.writer.add_scalar('Loss/{}'.format(phase), loss, epoch)
