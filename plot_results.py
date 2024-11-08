import matplotlib.pyplot as plt
import pandas as pd


def plot_results(input_file, output_file):
    # Load the CSV data
    data = pd.read_csv(input_file)

    # Plot the loss and validation loss over epochs
    plt.plot(data["Epoch"], data["Loss"], label="Training Loss")
    plt.plot(data["Epoch"], data["Validation Loss"], label="Validation Loss")

    # Add title and labels
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save the plot
    print(f"Saving plot to: {output_file}")
    plt.savefig(output_file)


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    plot_results(input_file, output_file)
