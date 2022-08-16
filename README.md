# Capstone Project (July-August 2022)

In this project we studied, trained, tested and deployed 5 Recommender Systems models to accomplish the tasks of candidate generation, scoring and ranking using the popular Amazon Reviews dataset. These models are:

| Collaborative Filtering  | Content-based Filtering | Hybrid   
| ------------------------ | ----------------------- | --------    |
| **S**hort-term and **L**ong-term<br>preference **I**ntegrated<br>**Rec**ommender (SLi-Rec) | LightGBM | **Wide & Deep** Learning<br>for Recommender<br>Systems |
| **S**elf-**A**ttentive **S**equential<br>**Rec**ommendation (SASRec) | | e**x**treme **Deep**<br>**F**actorization **M**achine (xDeepFM) |

In order to recreate this project, the following steps are required.
1. git clone https://github.com/microsoft/recommenders.git
2. git clone https://github.com/ss-github-code/capstone_recsys
3. Essentially you can start by downloading the 5-core dataset from [here](http://deepyeti.ucsd.edu/jianmo/amazon/index.html). Follow the data preparation steps laid out in the [report](https://github.com/ss-github-code/capstone_recsys/blob/main/report/report.md).
4. However, if you are a University of Michigan faculty or student, you can access the prepared datasets, transformed data, trained models (if you are not training these models from scratch) shared on the Google drive. This will save you a lot of time!
5. Once you have downloaded the data either from the link above or from the shared Google drive, run `pip install -r requirements.txt` to install the required packages.
6. If you are starting from the 5-core dataset, start by running the data preparation steps found in this [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/preprocessing/amzn_gen_dataset.ipynb).
7. In case you do have access to the prepared dataset, you can start in any of notebooks for the 5 models under study in the [modeling](https://github.com/ss-github-code/capstone_recsys/tree/main/modeling) folder.

For more information and details, refer to the [project report](https://github.com/ss-github-code/capstone_recsys/blob/main/report/report.md).
