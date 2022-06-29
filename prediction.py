import torch
import matplotlib.pyplot as plt
import numpy as np


def predict(image):
    model = torch.load('net.pkl')
    image = np.asarray(image).astype(np.float32)
    with torch.no_grad():
        image = torch.tensor(image)
        image = image.reshape(-1, 784)
        out = model(image)
        _, pred = torch.max(out, 1)
        return pred.item()