# Snakefile

import yaml
import os

# Load configurations from config.yaml
with open("config.yaml") as f:
    config_data = yaml.safe_load(f)

configs = config_data["parameters"]

# Define the final targets (all model plots and aggregated results)
rule all:
    input:
        expand("results/{config_id}/model_plot.png", config_id=[c["id"] for c in configs]),
        "results/aggregated_metrics.csv",
        "results/aggregated_plot.png"

# Rule to train the MNIST model for each configuration
rule train_model:
    input:
        script="train_mnist.py"
    output:
        model="results/{config_id}/model.h5",
        output_txt="results/{config_id}/model_output.txt"
    params:
        epochs=lambda wildcards: next(c for c in configs if c["id"] == wildcards.config_id)["epochs"],
        batch_size=lambda wildcards: next(c for c in configs if c["id"] == wildcards.config_id)["batch_size"]
    threads: 4
    shell:
        """
        mkdir -p results/{wildcards.config_id}
        if [ ! -f {output.model} ]; then
            echo "Training model for {wildcards.config_id} with epochs={params.epochs}, batch_size={params.batch_size}..."
            python {input.script} --epochs {params.epochs} --batch_size {params.batch_size} --output {output.model}
        else
            echo "Model for {wildcards.config_id} already exists. Skipping training."
        fi
        """

# Rule to convert the model output (TXT to CSV) for each configuration
rule convert_output:
    input:
        "results/{config_id}/model_output.txt"
    output:
        "results/{config_id}/model_output.csv"
    run:
        with open(input[0], 'r') as txt_file:
            data = txt_file.readlines()

        # Prepare to write to CSV
        with open(output[0], 'w') as csv_file:
            # Write CSV headers
            csv_file.write("Epoch,Loss,Validation Loss,Accuracy,Validation Accuracy\n")
            
            # Initialize placeholders for metrics
            epoch = None
            loss = None
            val_loss = None
            accuracy = None
            val_accuracy = None

            # Parse the output.txt file line by line
            for line in data:
                line = line.strip()

                # Detect epoch number
                if line.startswith("Epoch"):
                    if epoch is not None:
                        # Write the previous epoch's data to CSV
                        csv_file.write(f"{epoch},{loss},{val_loss},{accuracy},{val_accuracy}\n")
                    # Get the new epoch number
                    epoch = line.split()[1]
                
                # Detect loss, accuracy, and validation metrics
                elif line.startswith("Loss:"):
                    loss = line.split(":")[1].strip()
                elif line.startswith("Validation Loss:"):
                    val_loss = line.split(":")[1].strip()
                elif line.startswith("Accuracy:"):
                    accuracy = line.split(":")[1].strip()
                elif line.startswith("Validation Accuracy:"):
                    val_accuracy = line.split(":")[1].strip()

            # Write the last epoch's data to CSV
            if epoch is not None:
                csv_file.write(f"{epoch},{loss},{val_loss},{accuracy},{val_accuracy}\n")

# Rule to generate plots from the CSV for each configuration
rule plot_results:
    input:
        "results/{config_id}/model_output.csv"
    output:
        "results/{config_id}/model_plot.png"
    shell:
        "python plot_results.py {input} {output}"

# Rule to aggregate metrics across all configurations
rule aggregate_metrics:
    input:
        expand("results/{config_id}/model_output.csv", config_id=[c["id"] for c in configs])
    output:
        "results/aggregated_metrics.csv"
    run:
        import pandas as pd

        df_list = []
        for csv_file in input:
            config_id = os.path.basename(os.path.dirname(csv_file))
            df = pd.read_csv(csv_file)
            df["config_id"] = config_id
            df_list.append(df)

        aggregated_df = pd.concat(df_list, ignore_index=True)
        aggregated_df.to_csv(output[0], index=False)

# Rule to plot aggregated results
rule plot_aggregated_results:
    input:
        "results/aggregated_metrics.csv"
    output:
        "results/aggregated_plot.png"
    shell:
        "python plot_aggregated_results.py {input} {output}"
