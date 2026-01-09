# MLOps Project - Brain Tumor Segmentation

## Project description

Exam project for course 02476 Machine Learning Operations at Technical University of Denmark (DTU), Jan 2026. Created by group 65.

Authors: \
Lotte Alstrup, s204297 \
Rikke Alstrup, s194693 \
Mejse Grønborg-Koch, s196050

### Overall goal of the project

In this project we aim to use a deep learning model to segment brain tumors in images - specifically CT scans - of human heads. The overall goal of this project is to succesfully utilize machine learning operations (MLOps) practices throughtout the modelling process. 

### Third party framework 

We will use [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) as our third party framework. U-Net is convolutional neural network specially designed for image segmentation. nnU-Net is a framework which analyzes the given training data and configure a U-Net pipeline based specifically on the given dataset. Therefore, nnU-Net can easily adapt to different datasets. 

We will use nnU-Net to configure a U-Net model based on our dataset (see section below). This is the model we will train to segment brain tumors and hopefully fine-tune to enhance performance.

### Data 

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
