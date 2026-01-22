# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8) - NOTE: not done as in M8 but as in M21.
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

65


### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

*s204297, s196050, s194693*

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used nnU-Net as a third-party framework to support the development of a medical image segmentation algorithm for our project. nnU-Net provides a standardized datastructure and well default configurations for model architecture and the hyperparameters. We relied on nnU-Net’s built-in planning and preprocessing steps to adapt the model to the Hippocampus MRI dataset, as well as its predefined training and inference pipelines for model training and evaluation.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used ```conda``` and ```pip``` to manage packages and environments. Each group member created their own local clean conda environment for the project to help keep track of which packages were used specifically for the project. We kept track of Python dependencies through ```requirements.txt``` and ```requirements_dev.txt```, where we listed all Python packages as well as versions used in the project. 

If a new group member was to get an exact copy of our environment they would go through the following steps in the terminal:
1. Create a fresh ```conda``` environment using Python version 3.12: ```conda create -n mlops_project python=3.12```
2. Activate above environment: ```conda activate mlops_project```
3. Install the dependencies and the project itself using the ```requirement.txt``` file:
```txt 
python3 -m pip install -r requirements.txt 
python3 -m pip install -e .
```

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

From the cookiecutter templete the overall project structure was generated. Some of the adaption we made in our project was:

The ```​.devcontainer/``` folder configures the VS Code Dev Containers extension to ensure a reproducible development environment. The ​```.dvc/ directory``` is created during DVC initialization and contains configuration for data versioning.

We did not use the github agent dtu_mlops_agent.md

While the project scaffold included GitHub Actions support, the CI workflows located under ```​.github/workflows/``` were explicitly modified and consolidated by the team during development to better fit the project’s needs.

The ```data/``` directory follows a different structure than the one generated by the cookiecutter template. This deviation is necessary because nnU-Net requires a fixed directory layout to operate correctly. Specifically, nnU-Net expects the following top-level directories: ```nnUNet_raw/```, ```nnUNet_preprocessed/```, and ```nnUNet_results/```. The ```nnUNet_results/``` directory stores trained models, logs, and inference outputs after model training. Which is why the models.py is removed under ```src/```.

Within each of these directories the dataset is placed in its own subfolder, ```Dataset621_Hippocampus/```. In this project, ```the nnUNet_raw/``` and ```nnUNet_preprocessed/``` directories were created manually to match nnU-Net’s expected structure, whereas the ```nnUNet_results/``` directory is automatically created by nnU-Net during training.

During the preprocessing step, nnU-Net generates metadata files such as dataset_fingerprint.json, which capture dataset properties and are required for reproducible training and inference.

Under normal circumstances, nnU-Net expects the raw dataset to be located inside the ```nnUNet_raw/``` directory. In this project, the raw data has been uploaded directly to the cloud instead. This design choice is explained in further detail in question 10.

The Docker configuration files ```api.dockerfile``` and ```train.dockerfile``` were updated to reflect the final training and inference setup.

The ```docs/``` directory was initially part of the cookiecutter template, it was not used in the project due to time constrains. However, it could have been used to store extended technical documentation or user guides.

No Jupyter notebooks were used during the project, and the ```notebooks/``` directory was therefore removed.

Within ```src/mlops_project/```, the file ```visualize.py``` was removed because it is not used in the project. The file ```evaluate.py``` was also removed, as nnU-Net provides its own evaluation mechanisms. The files ```api.py```, ```data.py```, ```model.py```, and ```train.py``` were implemented, and ```data_drift.py``` was added to test for data drift. The output of this analysis is the ```drift_train_vs_test.html```which is located under the  ```report/```folder.

In the ```tests/``` directory, ```test_api.py``` is currently not fully implemented. The remaining tests have been implemented, and ```conftest.py``` was added to support shared test fixtures.

A ```cloudbuild.yaml``` file was added to define a Google Cloud Build pipeline for building and pushing Docker images.

The ```pytest.ini``` file was created when continuous integration (CI) was introduced. It configures the pytest test runner and defines custom test markers used in the project.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We implemented a linting step using ruff in the continous integration pipeline. This should check if the code has basic quality rules and is PEP8-compliant. If an error occured, we used the command ''ruff fix'' if possible or changing it manually. Hopefully this makes the code just a bit more readable, as this can be hard enough as is. 

We did not talk about any specific rules for typing and documentation. This is clear in the commenting throughout the code that we have our own way of doing this. When starting a new larger project, a clear guideline of this would defiently be preferred so firstly it is the same throughout the scripts as well as everyone agrees on what needs to be written in order to understand the code.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented nine tests in total. Eight unit tests focus on the data preprocessing pipeline and verify that helper functions correctly rename files, filter unwanted files, copy images and labels, and generate the expected nnU-Net v2 dataset structure. Initially, the tests assumed that the dataset had already been downloaded, but they were later refactored into pure unit tests so they run without requiring real data files. Additionally, one unit test was implemented for the model code to verify that preprocessing outputs from nnUNet are correctly added to the project’s configuration, which is the main responsibility of the model.py file.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of the project is 26%. This is because we did not write tests for all parts of the pipeline. In particular, the training code and the API code have 0% coverage, since no automated tests were created for these parts. The data processing code has a higher coverage of 73%, and the model code has a coverage of 45%, as these parts were easier to test with unit tests.

Even if the code coverage was close to 100%, we would not expect the code to be completely free of errors. Code coverage only shows how much of the code is run during testing, not whether all possible cases are tested. The code may still fail when using different data, unusual inputs, or in real-world situations. However, a high code coverage is still a good sign, because it shows that much of the code has been tested and helps increase confidence that the code works as expected.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

From the very beginning we used branches as a part of our workflow. This helped us work in parallel without accidently messing up each other's work. It also improved version control, since it is easier to pinpoint changes and exactly who made the changes. 

Everytime a group member was working on a new task they created a new branch dedicated to this task. Thus, many branches were created during the project. Our philosophy was to always create a new branch to make sure that our starting point for a task was up to date with the main branch, and then to commit and push our changes regularly, so other group members would be up to date with our work. 

For the first two weeks of the course we mostly merged our local branches with ```main``` locally and then pushed to ```main```, whereas in the last week we learned to use pull requests. With pull requests we could see if our changes failed or succeded the integrated GitHub tests before merging, and if needed we could request other group members to review the changes. 

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used Data Version Control (DVC) in the project through Google Cloud Platrform (GCP). We created a project called ```mlops-project-group65```, and in Google Cloud Storage (GCS) we added a bucket to store our data. We opted to store the original data downloaded from [Medical Segmentation Decathlon](http://medicaldecathlon.com/dataaws/) as well as the data in the folder ```nnUNet_raw```, which is processed to meet the input format for nnU-Net. With this we made sure that even if Medical Segmentation Decathlon changed the data on their website we still had the same version of our data in storage, which improves reproducibility. 

We did not store the preprocessed data, which is preprocessed as a part of the nnU-Net pipeline, nor any results. These datasets can easily be reproduced from the original data. 

The datasets tracked by DVC was ignored by Git, ie. not uploaded to GitHub. If a person wants to precisely replicate our project they need to both clone the Git repository and pull the data from GCS (access to the cloud project is required).

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We made use of ```hydra``` and a ```config.yaml``` file to configure experiments. In ```config.yaml``` we specified datasets, paths, model/training configurations and hyperparameters, and Weights and Biases (W&B) settings. In our ```train.py``` we used ```hydra``` to configure the settings from ```config.yaml```. To run an experiment you simply write ```python src/mlops_project/train.py``` in the terminal from the repo root. If you need to change for instance the batch size you change ```batch_size``` inside ```config.yaml```, save the config file, and run the experiment again.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Each time an experiment is run an instance of how the ```config.yaml``` looked at runtime is saved. This ensures that all information about the configuration of the experiment is logged and can be used to reproduce the experiment with the exact same configurations. 

In the config file we also defined a seed, which was set in ```train.py```, which improves reproducibility. 

Furthermore, we used ```loguru``` and ```wandb``` (W&B) to log the experiments. Logging using ```loguru``` creates local logs whereas ```wandb``` logs the experiments online on [wandb.ai](https://wandb.ai/site/). The W&B website can also be used to visualize and compare experiments.

Lastly, a log is created as part of the nnU-Net pipeline and saved together with nnU-Net results. 

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We managed to write a inference API using ```FastAPI```. The API loads our project settings from ```configs/config.yaml``` (via OmegaConf) so the dataset id (621), nnU-Net configuration (2d), fold, trainer, plans, and default device are controlled through one central config file instead of hardcoding. On startup we also set the required nnU-Net environment variables (nnUNet_raw, nnUNet_preprocessed, nnUNet_results) and create the directories if they do not exist.

We added a health endpoint ```(GET /)``` that reports the active configuration and whether a trained model is available by checking for ```checkpoint_final.pth```. The prediction endpoint (POST /predict) accepts a ```.nii```/```.nii.gz``` upload, writes it to a temporary nnU-Net-compatible input folder ```(case_0000.nii.gz)```, runs nnUNetv2_predict via subprocess, and returns the resulting segmentation mask as a downloadable NIfTI file. The endpoint also supports choosing cpu/cuda/mps through a query parameter.

Although the trained model was detected successfully, inference could not be completed due to package-level incompatibilities between PyTorch, NumPy, and nnU-Net in the local environment.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We deployed our inference API both locally and in the cloud. Locally, the API was served using FastAPI and Uvicorn and could be accessed via ```localhost```, where the service exposed a health endpoint and interactive Swagger UI. This confirmed that the API started correctly and detected the trained nnU-Net checkpoint.

For cloud deployment, the API was containerized using Docker and built and pushed via Google Cloud Build to Artifact Registry. The image was then deployed as a managed service on Google Cloud Run in the ```europe-west1``` region with public access enabled. The deployment succeeded, and the service was exposed through a public HTTP endpoint, verifying that the application started correctly in the cloud.

The deployed service can be invoked by sending a POST request to the /predict endpoint with a NIfTI image, for example using: ```curl -X POST "<SERVICE_URL>/predict" \  -F "image=@example_image.nii.gz"```

While the service was reachable both locally and in the cloud, full end-to-end inference could not be completed due to runtime dependency incompatibilities between ```PyTorch```, ```NumPy```, and ```nnU-Net```.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
