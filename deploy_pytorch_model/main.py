import torch
from torch.utils import data
from torch import nn

def synthetic_data(m, c, num_examples):
    X = torch.normal(0, 1, (num_examples, len(m)))
    y = torch.matmul(X, m) + c
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
 
 
true_m = torch.tensor([2, -3.4])
true_c = 4.2
features, labels = synthetic_data(true_m, true_c, 1000)



def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

net = nn.Linear(2, 1)
net.weight.data.normal_(0, 0.01)
net.bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 5
 
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward() 
        trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

m = net.weight.data
print('error in estimating m:', true_m - m.reshape(true_m.shape))
c = net.bias.data
print('error in estimating c:', true_c - c)