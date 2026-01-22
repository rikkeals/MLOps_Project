# MLOps Project - Hippocampus segmentation

## Project description

Exam project for course 02476 Machine Learning Operations at Technical University of Denmark (DTU), Jan 2026. Created by group 65.

Authors: \
Lotte Alstrup, s204297 \
Rikke Alstrup, s194693 \
Mejse Grønborg-Koch, s196050

### Overall goal of the project

In this project we aim to use a deep learning model to segment hippocampus left and right part in images - specifically MRI scans - of human heads. Segmentation is the process of identifying and outlining structures - in this case we aim to outline hippocampus in the brain.
The overall goal of this project is to succesfully utilize machine learning operations (MLOps) practices throughtout the modelling process. 

### Third party framework 

We will use [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) as our third party framework. U-Net is a convolutional neural network (CNN) specially designed for image segmentation. nnU-Net is a framework based on U-Net designed specifically for segmentation of medical images. Instead of designing a new neural network architecture for every new dataset, the nnU-Net framework analyzes the given dataset and configures a U-Net based pipeline suited to said dataset. nnU-Net automatically determines configurations such as normalization methods and loss functions as well as hyperparameters such as batch size and network depth, reducing the "startup" time when creating a new model for a new dataset. Therefore, nnU-Net can easily adapt to different datasets. 

We will use nnU-Net to configure a U-Net model based on our dataset (see section below). This is the model we will train to segment hippocampus and hopefully fine-tune to enhance performance.

### Data 

The dataset used in this project is [Task04 – Hippocampus from the Medical Segmentation Decathlon](http://medicaldecathlon.com/dataaws/). It consists of 3D MRI scans of human brains with corresponding voxel-level annotations of the left and right hippocampus.

The dataset contains a total of 394 3D MRI scans: 263 for training and 131 for testing. Each MRI volume is provided as a NIfTI file (.nii.gz) along with a segmentation mask where each voxel is labeled as background, left hippocampus, or right hippocampus. This makes the task a multiclass, voxel-wise segmentation problem.

The data is split into predefined training and test sets and is stored in the project’s data/raw directory before being processed and converted into the format required by nnU-Net.

### Deep learning model used

As mentioned above, the deep learning model used will be a U-Net model for image segmentation. 

## Project structure

The directory structure of the project looks like this:
```txt
├── .devcontainer/            # Devcontainer setup
│   ├── devcontainer.json     
│   └── post_create.sh        
├── .dvc/                     # Data version control setup
│   ├── .gitignore            
│   └── configs               # DVC configuration
├── .github/                  # Github actions and dependabot
│   ├── agents/               # We did not make use of the dtu_mlops_agent.md
│   ├── promts/
│   ├── workflows/
|       ├── ci.yaml           # Continous integration pipeline
|       ├── linting.yaml      # Code quality check
│       └── pre-commit-update.yaml
|   └── dependabot.yaml       # Automated dependency update configuration
├── configs/                  # Configuration files
├── data/                     # Data directory (with nnUNet data structure)
│   ├── nnUNet_preprocessed/.../
|       ├── gt_segmentation/  # ground truth labels
|       ├── nnUNetPlans_2d/   # created by nnUNet planning
|       ├── dataset_fingerprint.json # created by nnUNet preprocessing
|       ├── dataset.json      # manual setup file
|       ├── nnUNetPlans.json  # created by nnUNet planning
│       └── splits_final.json # created by nnUNet preprocessing
│   ├── nnUNet_results/       # nnUNet creates during training
│   ├── .gitignore            # Data not uploadet to github but to cloud
│   ├── nnUNet_raw.dvc        # Data tracking 
│   └── original.dvc          # Data tracking
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── outputs/                  # Outputs from Hydra and Loguru
│   └── folder with date/

├── reports/                  # Reports
│   ├── figures /
│   ├── drift_train_vs_test.html # Data drift summary
│   └── README.me             # Report and checklist
├── src/                      # Source code
│   ├── mlops_project/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data_drift.py
│   │   ├── data.py           # Provides correct datastructure
│   │   ├── model.py
│   │   └── train.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── AGENTS.md                 # We havn't used the agent
├── Cloudbuild.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── pytest.ini                # Pytest configuration and custom test markers
├── README.md                 # Project README
├── requirements_dev.txt      # Development requirements
├── requirements.txt          # Project requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## How to run the project

1. Clone GitHub repository (repo) locally 
2. Activate your conda environment
3. Run the requirement.txt file by typing 
```txt 
python3 -m pip install -r requirements.txt 
python3 -m pip install -e .
```
4. 

### Run using Docker 

The project can also be run using Docker by building and and executing a Docker image from the Docker file train.dockerfile. You need to [install Docker](https://docs.docker.com/get-started/get-docker/) to do this. In the terminal write the following (and make sure that your current directory is /MLOps_Project, i.e. the root of the project):

1. Build a Docker image: ```docker build -f dockerfiles/train.dockerfile . -t train:latest```
2. Execute the Docker image: ```docker run --rm --shm-size=2g -v $(pwd)/data:/data --name experiment1 train:latest```

Note: ```--shm-size=2g``` sets the shared memory to 2 GB in the Docker container (default is 64 MB). You can try without, but will most likely get the error message "No space left on device". This is because PyTorch and nnU-Net use multiprocessing and has workers that share large tensors. 

Note: [$(pwd) needs to change depending on your OS](https://stackoverflow.com/questions/41485217/mount-current-directory-as-a-volume-in-docker-on-windows-10).


## Data versioning and storage

*Section modified from / written with the help of ChatGPT*.

This project uses DVC (Data Version Control) together with Google Cloud Storage (GCS) to version and store datasets. The datasets are thus not tracked by Git, but by DVC instead.

The following directories are tracked with DVC:
- ```data/original``` – original dataset downloaded from the Medical Segmentation Decathlon
- ```data/nnUNet_raw``` – dataset converted to nnU-Net input format using data.py

The following directory is *not* tracked with DVC:
- ```data/nnUNet_preprocessed``` – dataset derived from running train.py

Only the corresponding .dvc metadata files are committed to Git.

### Pull data from Google Cloud

Prerequisites:
- A Google account
- Access to the Google Cloud Platform (GCP) project ```mlops-project-group65```
- Dependencies installed via ```requirements.txt```
- ```gcloud``` installed ([installation guide](https://docs.cloud.google.com/sdk/docs/install-sdk))
- This Git repo cloned locally
- Your cd set to the repo root in the terminal

Type in terminal:
1. Authenticate with Google Cloud: ```gcloud auth login```
2. Set active GCP project: ```gcloud config set project mlops-project-group65```
3. Pull data from cloud: ```dvc pull```

The datasets should be downloaded into data/ locally.

## Run tests on pipeline using Pytest
1. Run "pip install -e ." in the project root.
2. Run all tests by "pytest tests\"
   - If you only want to test one part of the pipeline specify this by e.g. "pytest tests\test_data.py"

## Run the FastAPI inference service
Type in terminal:
PYTHONPATH=src python -m uvicorn mlops_project.api:app --reload --port 8000

The API will be available at: http://localhost:8000
