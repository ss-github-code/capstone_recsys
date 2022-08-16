# Capstone Project (July-August 2022)

In this project we studied, trained, tested and deployed 5 Recommender Systems models to accomplish the tasks of candidate generation, scoring and ranking using the popular Amazon Reviews dataset. These models are:

| Collaborative Filtering  | Content-based Filtering | Hybrid   
| ------------------------ | ----------------------- | --------    |
| **S**hort-term and **L**ong-term<br>preference **I**ntegrated<br>**Rec**ommender (SLi-Rec) | LightGBM | **Wide & Deep** Learning<br>for Recommender<br>Systems |
| **S**elf-**A**ttentive **S**equential<br>**Rec**ommendation (SASRec) | | e**x**treme **Deep**<br>**F**actorization **M**achine (xDeepFM) |

## Setting up the environment
In order to recreate this project, the following steps are required.
1. git clone https://github.com/microsoft/recommenders.git
2. git clone https://github.com/ss-github-code/capstone_recsys
3. Essentially you can start by downloading the 5-core dataset from [here](http://deepyeti.ucsd.edu/jianmo/amazon/index.html). Download the All_Amazon_Meta.json.gz and All_Amazon_Review_5.json.gz. However, if you are a University of Michigan faculty or student, you can access the prepared datasets, transformed data, trained models (if you are not training these models from scratch) shared on the Google drive. This will save you a lot of time!
4. Once you have downloaded the data either from the link above or from the shared Google drive, run `pip install -r requirements.txt` to install the required packages.
5. If you are starting from the 5-core dataset, start by running the data preparation steps found in this [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_gen_dataset.ipynb).
6. There are several model specific input data requirements. In order to save time, we have preprocessed the output from step 5 and saved the intermediate results (and copied them to a shared Google drive folder).
7. If you want to recreate the model specific input data requirements, run the Jupyter notebooks in the [data preparation](https://github.com/ss-github-code/capstone_recsys/tree/main/data_preparation) folder in the following order (specific ordering is required in order to reuse the output of the data preparation steps from one notebook to the next):
  - First, run the notebook to prepare input data for the [Wide & Deep](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_gen_input_wide_deep.ipynb) model
  - Next, run the notebook to prepare input data for the [xDeepFM](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_gen_input_xdeepfm.ipynb) model
  - The notebooks for preparing data for [LightGBM](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_gen_input_lightgbm.ipynb) and [SLi-Rec and SASRec](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_gen_input_slirec.ipynb) models can be run next. 
8. In case you do have access to the prepared datasets, you can start in any of notebooks for the 5 models under study in the [modeling](https://github.com/ss-github-code/capstone_recsys/tree/main/modeling) folder.

## Train, validate, and test models
In general, each Jupyter notebook in the [modeling](https://github.com/ss-github-code/capstone_recsys/tree/main/modeling) folder follows the same set of steps. They are:
1. Any additional data preprocessing/preparation required
2. Load in the hyper-parameters
3. Load in the model
4. Train the model using the train and validation datasets, follow the training using Tensorboard. Generate plots for validation and training loss and any other metrics to help show progress.
5. Save the best model for use in model serving.
6. Load the best model and generate predictions for the test dataset. This outputs the pairwise metrics NDCG@10 and Hit@10.
7. Generate predictions for the user with the most reviews for all items unseen by the user. You can select a different user if you want to.
8. Save the prediction scores, titles, category for the selected user. The output generated in this step is the input to our dashboard where we analyze the difference in the recommendations made by the 5 models.

## Link to the project report
For more information and details about each model and the comparison between models, refer to the [project report](https://github.com/ss-github-code/capstone_recsys/blob/main/report/report.md).
