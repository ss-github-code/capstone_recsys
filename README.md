# Capstone Project (July-August 2022)

In this project we studied, trained, tested and deployed 5 Recommender Systems models to accomplish the tasks of candidate generation, scoring and ranking using the popular Amazon Reviews dataset. These models are:

| Collaborative Filtering  | Content-based Filtering | Hybrid   
| ------------------------ | ----------------------- | --------    |
| **S**hort-term and **L**ong-term<br>preference **I**ntegrated<br>**Rec**ommender (SLi-Rec) | LightGBM | **Wide & Deep** Learning<br>for Recommender<br>Systems |
| **S**elf-**A**ttentive **S**equential<br>**Rec**ommendation (SASRec) | | e**x**treme **Deep**<br>**F**actorization **M**achine (xDeepFM) |

In order to recreate this project, the following steps are required.
1. git clone https://github.com/microsoft/recommenders.git
2. git clone https://github.com/ss-github-code/capstone_recsys
3. Use the Jupyter notebooks in the modeling folder of this repository. Note that you will need access to the datasets, transformed data, trained models (if you are not interested in training these models from scratch) shared on the Google drive.
4. Once the shared Google drive has been mounted and if you are running on Google colab, most of the package requirements are already met (including TensorFlow, Pandas, Numpy). However, requirements.txt is provided for a full list of packages if you want to recreate the Conda environment on a local machine.

For more information and details, refer to the [project report](https://github.com/ss-github-code/capstone_recsys/blob/main/report/report.md).
