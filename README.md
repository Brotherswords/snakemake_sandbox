# Snakemake Workflow Sandbox

## Overview

This Snakemake workflow facilitates the training of MNIST models under various configurations, supporting both local execution and submission to High-Performance Computing (HPC) environments using `sbatch`. It is designed to handle multiple configurations seamlessly, allowing efficient resource management and avoiding redundant computations.

## Features

- **Multiple Configurations Management**: Train multiple model configurations defined in a single `config.yaml` file without rerunning already completed jobs.
- **Local Execution**: Run all jobs locally on your machine for development and testing purposes.
- **HPC Submission**: Submit jobs to an HPC cluster using `sbatch` with customizable resource parameters.
- **Dynamic Resource Allocation**: Specify resource requirements (e.g., CPU cores, memory, time) per job configuration.
- **Snakemake Profiles Integration**: Utilize Snakemake profiles to manage execution environments efficiently, separating concerns out of the Snakefile (In Progress)
- **Automated Job Monitoring**: Automatically monitor and wait for HPC jobs to complete before proceeding to downstream tasks.
- **Output Management**: Organized results including trained models, logs, CSV metrics, and plots per configuration.
- **Scalable/Reusable**: Easily extendable for additional configurations and adaptable to different computational environments.
