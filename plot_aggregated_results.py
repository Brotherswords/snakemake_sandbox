# plot_aggregated_results.py

import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_aggregated_results(input_csv, output_plot):
    # Load the aggregated metrics
    data = pd.read_csv(input_csv)

    # Example: Plot final epoch accuracy for each configuration
    # Adjust the plotting logic based on the structure of aggregated_metrics.csv
    final_metrics = (
        data.groupby("config_id")
        .agg(
            {
                "Epoch": "max",
                "Loss": "last",
                "Validation Loss": "last",
                "Accuracy": "last",
                "Validation Accuracy": "last",
            }
        )
        .reset_index()
    )

    # Create a bar plot for final accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(final_metrics["config_id"], final_metrics["Accuracy"], color="skyblue")
    plt.xlabel("Configuration ID")
    plt.ylabel("Final Accuracy")
    plt.title("Final Accuracy per Configuration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_aggregated_results.py <input_csv> <output_plot>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_plot = sys.argv[2]
    plot_aggregated_results(input_csv, output_plot)
