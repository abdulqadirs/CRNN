import torch.optim as optim

def sgd_optimizer(model):
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    return optimizer