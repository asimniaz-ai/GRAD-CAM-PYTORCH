import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import cv2

#device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# dataset of 1 image
dataset = datasets.ImageFolder(root='./data', transform=transform)

# dataloader for the dataset
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)


# define VGG class
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # pretrained vgg19
        self.vgg = models.vgg19(pretrained=True)
        # dissecting network upto the last conv layer
        self.features_conv = self.vgg.features[:36]
        # Apply Max Pool
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # get classifier
        self.classifier = self.vgg.classifier

        # placeholders for gradients
        self.gradients = None

    # Hook for gradients of activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # conv layers
        x = self.features_conv(x)
        # register the hook
        y = x.register_hook(self.activations_hook)
        # apply remaining pooling layer
        x = self.max_pool(x)
        # before classification layer we need to change dimensions of the preceding tensor
        x = x.view((1, -1))
        # feed to the classifier
        xx = self.vgg.classifier(x)

        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


vgg = VGG()
vgg.eval()

# get image from dataloader
img, _ = next(iter(dataloader))

# get the most likely prediction of the model
pred_class = vgg(img).argmax(dim=1).numpy()[0]
pred = vgg(img)

pred[:, pred_class].backward()

# Pull gradients out of the model
gradients = vgg.get_activations_gradient()
# pool the gradients across channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get activations of last onvolutional layer
activations = vgg.get_activations(img).detach()

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of activations
heatmap = torch.mean(activations, dim=1).squeeze()

# Normalize the heatmap
heatmap /= torch.max(heatmap)
heatmap = heatmap.numpy()

img = cv2.imread('./data/elephant/elephant.jpg')
heatmap_1 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap_1 = np.uint8(255 * heatmap_1)
heatmap_2 = heatmap_1
heatmap_1 = cv2.applyColorMap(heatmap_1, cv2.COLORMAP_JET)
superimposed_img = heatmap_1 * 0.4 + img
cv2.imwrite('./map.jpg', superimposed_img)  ###saves gradcam visualization image

#Threshold the heatmap to binary
heatmap_thresholded = (heatmap_2 > 0.5).astype(np.uint8)
#Find contours of the binary heatmap
contours, _ = cv2.findContours(heatmap_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Fit a rectangle around the contours
bounding_box = None
if len(contours) > 0:
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    bounding_box = box

# Draw the bounding box on the original image
if bounding_box is not None:
    cv2.drawContours(img, [bounding_box], 0, (0, 0, 255), 2)

# Save the resulting image with the bounding box
cv2.imwrite('result.jpg', img)