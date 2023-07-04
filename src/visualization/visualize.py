import click
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from src.models.model import CNN


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("data_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, data_filepath, output_filepath):
    """Runs visualizations"""
    logger = logging.getLogger(__name__)
    logger.info("initializing visualizations")

    # import model
    model = CNN()
    model.load_state_dict(torch.load(input_filepath + "/model.pt"))

    # FEATURE MAPS
    model_weights = []
    conv_layers = []
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")

    # retrieve a single image from the test dataset
    data = torch.load(data_filepath + "/testloader.pt")
    dataiter = iter(data)
    images, labels = next(dataiter)
    image = images[0]

    # store outputs
    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))

    # print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split("(")[0], fontsize=30)
    plt.savefig(output_filepath + "/feature_maps.png", bbox_inches="tight")

    # PLOTTING t-SNE REPRESENTATION OF THE EMBEDDINGS
    test_imgs = torch.zeros((0, 1, 28, 28), dtype=torch.float32)
    test_predictions = []
    test_targets = []
    test_embeddings = torch.zeros((0, 10), dtype=torch.float32)

    for x, y in data:
        embeddings, preds = model(x)

        test_predictions.extend(preds.detach().cpu().tolist())
        test_targets.extend(y.detach().cpu().tolist())
        test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), dim=0)
        test_imgs = torch.cat((test_imgs, x.detach().cpu()), dim=0)

    test_imgs = np.array(test_imgs)
    test_embeddings = np.array(test_embeddings)
    test_targets = np.array(test_targets)
    test_predictions = np.array(test_predictions)

    # init tsne
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)

    # save figure
    cmap = cm.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 10
    for lab in range(num_categories):
        indices = np.where(test_targets == lab)
        ax.scatter(
            tsne_proj[indices, 0],
            tsne_proj[indices, 1],
            c=np.array(cmap(lab)).reshape(1, 4),
            label=lab,
            alpha=0.5,
        )
    ax.legend(fontsize="large", markerscale=2)
    plt.savefig(output_filepath + "/tsne.png", bbox_inches="tight")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
