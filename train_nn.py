from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt


# device config
device = torch.device('cpu')

# hyperparameters
input_size = 784 
hidden_size = 100
n_classes = 10
n_epochs = 3
batch_size = 100
learning_rate = 0.001

# MNIST

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                            transform=transform.ToTensor(), 
                                            download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                            transform=transform.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)                                    

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap='gray')

plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, n_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, n_classes=n_classes)

# model = torch.nn.Sequential(
#     nn.Linear(input_size, hidden_size),
#     nn.ReLU(),
#     nn.Linear(hidden_size, n_classes)
# )
# loss and optimizer


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop

n_total_steps = len(train_loader)
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 784).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1 % 100) == 0:
            print(f'epoch {epoch+1}/{n_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 784).to(device)
        labels = labels.to(device)  
        outputs = model(images)
        # value, index
        _, pred = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (pred == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')


torch.save(model, 'net.pkl')
