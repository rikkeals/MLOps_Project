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

1. Activate your conda environment
2. Run the requirement.txt file by typing "python3 -m pip install -r requirements.txt
python3 -m pip install -e ."
3. 