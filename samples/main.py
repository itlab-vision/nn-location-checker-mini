from pathlib import Path
from sys import path as sys_path

src_directory = (
    Path("./src/").absolute() if Path("./src/").exists() else Path("../src/").absolute()
)
sys_path.append(str(src_directory))

import torch
import torch.nn as tnn
import torchvision.transforms.v2 as tt2  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import DataLoader

from build_cnn import CNnetwork
from dataset import Dataset
from json_loader import ModuleLoader
from model_segment import ModelSegment, SupportedModels
from utils import TensorShape


def main():
    dataset = Dataset("./dataset/", tt2.Compose([tt2.Resize((227, 227))]))

    loader = DataLoader(
        dataset,
        64,
        True,
    )

    alexnet_classifier_loader = ModuleLoader("./json_modules/alexnet_classifier.json")

    alexnet_part = ModelSegment(SupportedModels.ALEXNET, 2)

    if isinstance(out := alexnet_part.compute_shape(TensorShape(3, 227, 227)), int):
        raise ValueError("Must be TensorShape")
    else:
        output = out

    alexnet_classifier = alexnet_classifier_loader.load(output)

    cnn_model = CNnetwork(alexnet_part, alexnet_classifier)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(device)

    learning_rate = 0.1
    num_epochs = 10

    loss_function = tnn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate)

    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loader):
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
            outputs = cnn_model(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
