import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import torch
import torch.nn as tnn
import torchvision.transforms.v2 as tt2  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import DataLoader
from torchinfo import summary

from classification_network import ClassificationNetwork
from dataset import Dataset
from json_loader import ModuleLoader
from model_segment import ModelSegment, SupportedModels
from output_transforms import get_accuracy
from utils import TensorShape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(
    epochs: int,
    loader: DataLoader,
    device: torch.device,
    loss_function: tnn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    model: ClassificationNetwork,
) -> None:
    for epoch in range(epochs):
        logger.info(f"Epoch number {epoch} starts")
        for _, (images, labels) in enumerate(loader):  # pyright: ignore[reportAny]
            images = images.requires_grad_().to(device)  # pyright: ignore[reportAny]
            labels = labels.to(device)  # pyright: ignore[reportAny]
            outputs = model(images)  # pyright: ignore[reportAny]
            loss = loss_function(outputs, labels)  # pyright: ignore[reportAny]

            optimizer.zero_grad()
            loss.backward()  # pyright: ignore[reportAny]
            _ = optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        accuracy, _ = get_accuracy(loader, model, device)
        logger.info(f"Epoch number {epoch} ends")
        logger.info(f"Epoch accuracy: {accuracy}")
        logger.info(f"Epoch loss: {loss.item()}")


def main() -> None:
    dataset = Dataset("./dataset/", tt2.Compose([tt2.Resize((227, 227))]))
    batch_size = 64
    training_loader = DataLoader(
        dataset,
        batch_size,
        True,
    )

    test_loader = DataLoader(dataset, batch_size, False)

    alexnet_classifier_loader = ModuleLoader("./json_modules/alexnet_classifier.json")

    alexnet_part = ModelSegment(SupportedModels.ALEXNET, 2)

    if isinstance(out := alexnet_part.compute_shape(TensorShape(3, 227, 227)), int):
        raise ValueError("Must be TensorShape")
    output = out

    alexnet_classifier = alexnet_classifier_loader.load(output)

    classification_model = ClassificationNetwork(alexnet_part, alexnet_classifier)

    model_summary = summary(classification_model, verbose=0, depth=5, col_names=[])

    logger.info(f"\n{format_torchsummary(str(model_summary))}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classification_model = classification_model.to(device)

    learning_rate = 0.1
    loss_function = tnn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classification_model.parameters(), lr=learning_rate)
    num_epochs = 2

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of batches: {len(training_loader)}")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate {learning_rate:.4f}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Loss function: {loss_function.__class__.__name__}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")

    logger.info("Start of training")
    train(
        num_epochs,
        training_loader,
        device,
        loss_function,
        optimizer,
        classification_model,
    )
    logger.info("End of training")

    logger.info("Start of testing")
    accuracy, avg_time_per_image = get_accuracy(
        test_loader, classification_model, device
    )
    logger.info(f"Testing accuracy: {accuracy:.4f}")
    logger.info(f"Average time per image: {avg_time_per_image:.4f} ms")
    logger.info(f"Classification speed: {1 / avg_time_per_image:.4f} images/s")
    logger.info("End of testing")


def format_torchsummary(summary: str) -> str:
    previous = 0
    model_structure_start = 0
    for _ in range(3):
        model_structure_start = summary.find("\n", previous) + 1
        previous = model_structure_start

    model_structure_end = summary.find("\n", model_structure_start + 1)
    previous = model_structure_end
    while summary[model_structure_end - 1] != "=":
        previous = model_structure_end
        model_structure_end = summary.find("\n", model_structure_end + 1)

    return summary[model_structure_start:previous]


def configure_logger() -> None:
    to_logs_folder = Path("./logs/")
    if not to_logs_folder.exists():
        raise RuntimeError("Please, init logs folder in root of the project")
    file_handler = TimedRotatingFileHandler(
        to_logs_folder.joinpath("latest.log"), "midnight", encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    format_template = "%(asctime)s %(levelname)s:%(message)s"
    date_template = "%d/%m/%Y %H:%M:%S"
    logging.basicConfig(
        format=format_template,
        datefmt=date_template,
        handlers=[console_handler, file_handler],
    )


if __name__ == "__main__":
    configure_logger()
    main()
