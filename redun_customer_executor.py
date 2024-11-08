import subprocess
import os
from redun import task


# Define a Redun task to submit the SLURM job
@task()
def submit_slurm_job(output_file: str, epochs: int, batch_size: int) -> str:
    """
    Submits a SLURM job for training the model with TensorFlow.
    Args:
        output_file (str): Path to save the model.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
    Returns:
        str: Path to the model output file (after the SLURM job is completed).
    """

    # Path to your model script (adjust if necessary)
    model_script_path = "/path/to/model.py"

    # Create a SLURM job script (as a string)
    slurm_job_script = f"""
    #!/bin/bash
    #SBATCH --job-name=tf_model_training
    #SBATCH --output=slurm_output_%j.log
    #SBATCH --ntasks=1
    #SBATCH --gres=gpu:1           # Request 1 GPU (adjust as needed)
    #SBATCH --mem=8G               # Memory request
    #SBATCH --time=02:00:00        # Time limit (adjust as needed)
    #SBATCH --partition=gpu        # Use the GPU partition (adjust based on your HPC environment)

    module load python/3.8         # Load necessary modules
    module load tensorflow/2.6.0   # Load TensorFlow module

    # Activate virtual environment if necessary
    # source /path/to/virtualenv/bin/activate

    # Run the model training
    python {model_script_path} --output {output_file} --epochs {epochs} --batch_size {batch_size}
    """

    # Save the SLURM script to a file
    slurm_script_path = "submit_model_training.slurm"
    with open(slurm_script_path, "w") as f:
        f.write(slurm_job_script)

    # Submit the job to SLURM using subprocess
    slurm_command = ["sbatch", slurm_script_path]
    result = subprocess.run(slurm_command, capture_output=True, text=True)

    # Capture SLURM job ID from the output
    slurm_output = result.stdout
    job_id = slurm_output.strip().split()[-1]

    print(f"Job submitted to SLURM with job ID: {job_id}")

    # Optionally, wait for job to finish (poll job status) and return the output path
    # In this example, we just return the output file path for later processing
    return output_file
