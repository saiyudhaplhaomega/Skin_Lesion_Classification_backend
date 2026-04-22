"""
ResNet50 classifier wrapper for skin lesion classification.
Used by all RQ notebooks (RQ1-RQ6) for CAM generation.
"""
import torch
import timm


def create_model(model_name: str = 'resnet50', num_classes: int = 1, pretrained: bool = True):
    """
    Create a timm model for binary classification.

    Args:
        model_name: timm model name (default: resnet50)
        num_classes: number of output classes (1 for binary sigmoid)
        pretrained: load ImageNet pretrained weights

    Returns:
        timm model
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def get_target_layer(model, model_name: str = 'resnet50'):
    """
    Get the last convolutional layer for CAM generation.

    Args:
        model: the PyTorch model
        model_name: model architecture name

    Returns:
        target layer for CAM methods
    """
    if 'resnet' in model_name.lower():
        return model.layer4[-1]
    elif 'efficientnet' in model_name.lower():
        return model.blocks[-1][-1]
    elif 'mobilenet' in model_name.lower():
        return model.blocks[-1][-1]
    elif 'densenet' in model_name.lower():
        return model.features.denseblock4
    else:
        raise ValueError(f"Unsupported model: {model_name}. Add target layer manually.")