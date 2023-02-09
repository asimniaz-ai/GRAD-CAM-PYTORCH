# GRAD-CAM-PYTORCH

This repository is the implementation of GRAD-CAM with VGG19 in PyTorch framework. The reference paper is [Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).

This code is an implementation of Grad-CAM (Gradient-weighted Class Activation Mapping), a method for visualizing the regions of an image that contribute most to the prediction of a deep learning model. Grad-CAM uses the gradient information of the final prediction with respect to the activations of the last convolutional layer to create a heatmap highlighting the most important regions in the image.

The code uses the pre-trained VGG19 model from the torchvision library and creates a custom class VGG which inherits from nn.Module and disects the network to its last convolutional layer (layer 36). The custom class defines a hook function activations_hook to store the gradients of the activations during the backward pass and a method get_activations_gradient to retrieve the gradients. The method get_activations is used to extract the activations of the last convolutional layer.

A single image is loaded from the ./data/ directory using a data loader and passed through the model to get its prediction. The gradients are computed using the backward method of the final prediction and the activations and gradients are used to compute the heatmap using the Grad-CAM algorithm. The heatmap is superimposed on the original image using OpenCV and the result is displayed using Matplotlib.

## Brief detail and Steps to Follow

GRAD-CAM is a technique for visualizing the important regions in an image that contributed the most to a specific prediction made by a deep learning model. Here are the steps to implement GRAD-CAM in PyTorch:

- Load a pre-trained model: Load a pre-trained deep learning model in PyTorch, such as VGG, ResNet, or DenseNet.

- Define the target class: Specify the class of interest for which you want to generate a heatmap.

- Compute the gradients: Pass an input image through the model and compute the gradients of the target class with respect to the feature maps of the last convolutional layer.

- Global average pooling: Apply global average pooling to the computed gradients to obtain a weight matrix.

- Generate a heatmap: Use the weight matrix to weight the feature maps and sum them up to obtain a heatmap. This heatmap represents the importance of each location in the image with respect to the target class.

- Overlay the heatmap on the input image: Normalize the heatmap and overlay it on the input image to visually indicate the important regions.

- Create bounding boxes using opencv methods over the detected class.

- Save the result: Save the resulting heatmap and the overlaid image to visualize the important regions in the input image that contributed the most to the prediction of the target class.

