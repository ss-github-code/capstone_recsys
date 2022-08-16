# Capstone Project (July-August 2022)

In this project we studied, trained, tested and deployed 5 Recommender Systems models to accomplish the tasks of candidate generation, scoring and ranking using the popular Amazon Reviews dataset. These models are:

| Collaborative Filtering  | Content-based Filtering | Hybrid   
| ------------------------ | ----------------------- | --------    |
| **S**hort-term and **L**ong-term<br>preference **I**ntegrated<br>**Rec**ommender (SLi-Rec) | LightGBM | **Wide & Deep** Learning<br>for Recommender<br>Systems |
| **S**elf-**A**ttentive **S**equential<br>**Rec**ommendation (SASRec) | | e**x**treme **Deep**<br>**F**actorization **M**achine (xDeepFM) |

In order to recreate this project, the following steps are required.
1. git clone https://github.com/microsoft/recommenders.git
2. git clone https://github.com/ss-github-code/capstone_recsys
3. Use the Jupyter notebooks in the modeling folder of this repository. Note that you will need access to the datasets, transformed data, trained models (if you are not training these models from scratch) shared on the Google drive.
4. Once the data from the shared Google drive can be accessed, run `pip install -r requirements.txt` to install the required packages.

For more information and details, refer to the [project report](https://github.com/ss-github-code/capstone_recsys/blob/main/report/report.md).
