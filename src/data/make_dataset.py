# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed/).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ]
    )
    trainset = datasets.MNIST(
        input_filepath, download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.MNIST(
        input_filepath, download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # save loaders to output file_path
    torch.save(trainloader, output_filepath + "/trainloader.pt")
    torch.save(testloader, output_filepath + "/testloader.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
