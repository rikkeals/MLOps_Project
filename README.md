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
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
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
- This Git repo cloned locally
- Your cd set to the repo root in the terminal

Type in terminal:
1. Set active GCP project: ```gcloud config set project mlops-project-group65```
2. Create service account: ```gcloud iam service-accounts create dvc-sa-<yourname> \
  --display-name "DVC service account (<yourname>) ```
3. Grant access to cloud storage: ```gcloud projects add-iam-policy-binding mlops-project-group65 \
  --member="serviceAccount:dvc-sa-<yourname>@mlops-project-group65.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin```
4. Create a key file (OBS do NOT commit this file to Git): ```gcloud iam service-accounts keys create dvc-key.json \
  --iam-account=dvc-sa-<yourname>@mlops-project-group65.iam.gserviceaccount.com```
5. Authenticate locally: ```export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/dvc-key.json```
3. Pull data from cloud: ```dvc pull```

The datasets should be downloaded into data/ locally.



