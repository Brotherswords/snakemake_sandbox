# Rule to train the MNIST model
rule all:
    input:
        "results/model_plot.png"

rule train_model:
    threads: 4  # Allocate 4 threads (cores) to this rule
    input:
        script="train_mnist.py"  # Track changes in the training script -> Re-train if there are changes
    output:
        "results/model.h5",  # Saving the trained model
        "results/model_output.txt"  # Saving the model metrics
    params:
        epochs = 11,
        batch_size = 128,
        # force_retrain = False  # Option to force retraining
    shell:
        """
        # Convert Python boolean to bash-style condition
        if [ ! -f {output[0]} ]; then
            echo "Retraining model..."
            python train_mnist.py --epochs {params.epochs} --batch_size {params.batch_size} --output {output[0]}
        else
            echo "Model already exists. Skipping training."
        fi
        """


# Rule to convert the model output (TXT to CSV)
# Rule to convert the model output (TXT to CSV)
rule convert_output:
    input:
        "results/model_output.txt"
    output:
        "results/model_output.csv"
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


# Rule to generate plots from the CSV
rule plot_results:
    input:
        "results/model_output.csv"
    output:
        "results/model_plot.png"
    shell:
        "python plot_results.py {input} {output}"

