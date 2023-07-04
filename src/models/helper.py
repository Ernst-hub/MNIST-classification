import matplotlib.pyplot as plt


def save_loss_plot(data_list, file_path):
    epochs = [item[0] for item in data_list]
    losses = [item[1] for item in data_list]

    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")

    plt.savefig(file_path)
    plt.close()


if __name__ == "__main__":
    data_list = [[1, 2], [2, 4], [3, 6], [4, 8]]
    save_loss_plot(data_list, "test.png")
