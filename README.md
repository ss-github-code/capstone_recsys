# Capstone Project (July, August 2022)

## Problem Statement
### [Recommendations: What and Why](https://developers.google.com/machine-learning/recommendation/overview)
- Recommending the top k items to a user on their homepage
- Recommending the related set of k items once a user has clicked on a product

One common pipeline for Recommendation Systems consists of the following components:
- Candidate Generation
- Scoring
- Ranking

#### Candidate Generation
The system starts with a huge corpora and builds and trains a model in order to output a score to an item based on features provided as an input to the model. The exact nature of features depends on the type of model chosen.

There are two main approaches to candidate generation:
- **Collaborative Filtering**: Item and user similarity is used simultaneously to provide recommendations.
- **Content-based Filtering**: Item similarity is used to recommend items similar to what the user has liked before.

List of models considered in this study for candidate generation:

| Collaborative Filtering  | Content-based Filtering | Hybrid   
| ------------------------ | ----------------------- | --------    |
| **S**hort-term and **L**ong-term<br>preference **I**ntegrated<br>**Rec**ommender (SLi-Rec) | LightGBM | **Wide & Deep** Learning<br>for Recommender<br>Systems |
| **S**elf-**A**ttentive **S**equential<br>**Rec**ommendation (SASRec) | | e**x**treme **Deep**<br>**F**actorization **M**achine (xDeepFM) |

#### Scoring
After training, the model is used to generate a score for each unseen item for a given user.
#### Ranking
The generated scores are sorted in the descending order and the top k items are provided as recommendations to the user. The top k items are across several categories. We can also provide the top k items for a related category.

## Amazon Reviews Dataset
[Amazon Reviews](http://deepyeti.ucsd.edu/jianmo/amazon/index.html) dataset has 157+ million reviews, and 15+ million items.

We took the following steps to reduce the size of data (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/preprocessing/amzn_gen_dataset.ipynb)):
- Filter only those reviews that are from verified users and the item reviewed has meta information (left with 133+ million reviews)
- Next, recursively filter so that each user has reviewed at least 20 items and each item has been reviewed by 20 users (left with 38+ million reviews)
- Clean up meta information, consolidate main categories
- Finally only look at reviews of items whose main category and a selected list of subcategories belong to “Electronics” (includes 'Amazon Devices', 'Apple Products', etc.) (5.6+ million reviews from 830k+ users and 63k+ items in 36 categories as shown below).

| Key stats for the dataset | |
| -------------------------- | ----- |
| # reviews | 5,613,183 |
| # users | 830,668 |
| # items | 63,725 |
| # categories | 36 |

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/item_dist.jpg?raw=true" target=”_blank” alt="Log count of items vs category"/>

- For LightGBM, Wide & Deep, and xDeepFM, we consider both the main category as well as the sub-categories shown above.
- For SLi-Rec and SASRec, we only consider the main category as the code for the models do not have support for sub-categories. The item distribution for the 12 main categories is shown below.
<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/item_dist_main.jpg?raw=true" target=”_blank” alt="Log count of items vs main category"/>

- Chronological Split into train, validation, and test datasets. While the sequential models (SLi-Rec, SASRec) have an elaborate strategy to split the data chronologically into train, validation, and test datasets (the last record in the chronological sequence of reviews goes to the test, the second last to the validation, and the remaining to the train), we had to deploy the same strategy for the 3 models. We used the `python_chrono_split` from the [Microsoft Recommenders](https://github.com/microsoft/recommenders) framework that includes stratification and is available [here](https://github.com/microsoft/recommenders/blob/main/recommenders/datasets/python_splitters.py).
- Besides the chronological split, xDeepFM requires the data to be in Field-aware Factorization Machine (FFM) format where each row in the dataset has the following format: `<label> <field_index_id>:<feature_index_id>:<feature_value>`. (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/preprocessing/amzn_ffm.ipynb))

## Model Architecture
### LightGBM: A Highly Efficient Gradient Boosting Decision Tree

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_lightgbm.png?raw=true" alt="The architecture of LightGBM"/>

Notes about the model:<br>
- Model with fast training speed and high efficiency
- Support for parallel, distributed, and GPU learning
- We only require ordinal encoder to encode string like categorical features.

### [Wide & Deep](https://arxiv.org/abs/1606.07792): Wide & Deep Learning for Recommender Systems

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_widendeep.png?raw=true" alt="The architecture of Wide & Deep"/>

Notes about the model:<br>
- Memorization and Generalization: The model **memorizes** the correlation available in data due the frequent co-occurrence of items or features. However, correlations are transitive; so we need to **generalize** in order to discover new feature combinations that have never or rarely occurred in the past.<br>
- **Wide** component is a linear model (y = w<sup>T</sup>x + b); here x are raw input features and **cross-product transformations**, w = weights, and b = bias.<br>
- **Deep** component is a feed-forward neural network; where the inputs are categorical features converted into embedding vectors. Each hidden layer performs the following computation where l = layer, f = activation function (ReLU), a(l), b(l), and w(l) are activations, bias, and weights at l-th layer: a<sup>(l+1)</sup> = f(w<sup>(l)</sup>a<sup>(l)</sup> + b<sup>(l)</sup>)<br>

### [xDeepFM](https://arxiv.org/abs/1803.05170): Combining Explicit and Implicit Feature Interactions for Recommender Systems

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_xdeepfm.png?raw=true" alt="The architecture of Wide & Deep"/>

Notes about the model:<br>
- It shares some of the same attributes as the Wide & Deep model. However, unlike the previous model, xDeepFM does not require manual feature engineering for cross product transformations.
- **Compressed Interaction Network** (CIN) is used to generate cross feature interactions in an explicit manner. The details of CIN are quite similar to those of Convolutional Neural Networks (CNN). CIN lets the neural network learn cross feature interactions.
- The model combines CIN with a classical deep neural network (just like the **Deep** in Wide & Deep).
- Requires data according to the format required. Each row has a label (rating), and tab separated field_id:feature_id:feature_value for both numeric (user id, item id) and categorical (category/sub-category) features.<br>

### [SLi-Rec](https://www.microsoft.com/en-us/research/uploads/prod/2019/07/IJCAI19-ready_v1.pdf): Adaptive User Modeling with Long and Short-Term Preferences for Personalized Recommendation

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_slirec.png?raw=true" alt="The architecture of SLi-Rec"/>

Notes about the model:<br>
- The model focuses on solving two key problems: **dynamic time intervals** and **dynamic latent intent**. Typical user interactions have dynamic time intervals, e.g. consider two actions 5 minutes apart vs. two actions days apart. This kind of temporal distance deserve special handling. The model uses an upgraded Long Short-Term Memory (LSTM) layer called Dynamic RNN as shown above.
- In addition, user’s intent is also dynamic and changes from session to session. Irrelevant actions a useless for predicting a user's future action. E.g. suppose a user's review history is (iPhone-xs, airpods, cat food), when we want to recommend laptops (Macbook), only the first two actions make sense. To cope with content-aware distance, the authors use an attention mechanism (ATTN FCN) as shown above.
- In order to capture the static components influencing users' behaviors, which reflect their long-term behavior, the authors adopt the attentive "Asymmetric SVD" paradigm.
- Finally, the model uses an attention based fusion method to adapt to short and long term preferences:
  - **When**: if next action occurs shortly after the last action; short-term information plays a major role in prediction; otherwise long-term component weighs more.
  - **What**: if recent actions share a distinct intent/preference, then the next action may have a higher probability to share the same intent. (iPhones, airpods,.... MacBook?)<br>

### [SASRec](https://arxiv.org/abs/1808.09781): Self-Attentive Sequential Recommendation

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_sasrec.png?raw=true" alt="The architecture of SASRec"/>

Notes about the model:<br>
- Unlike existing sequential models that use convolutional or recurrent modules, SASRec uses Transformer based on the 'self-attention' mechanism.
- This is the same mechanism used in Natural Language Processing that has proven to be highly effective in uncovering complex syntactic and semantic patterns between words in a sentence.
- Since self-attention model does not include any recurrent or convolutional module, it is not aware of the positions of previous items. Hence, the authors inject a learnable position embedding layer.
- In order to tackle the problems of overfitting and vanishing gradients, the authors use both dropout and residual connections as shown above.

## Modeling
We used the [Microsoft Recommenders](https://github.com/microsoft/recommenders) for this study. The repository offers implementations of the above models and provides examples and best practices for building recommendation systems.

### LightGBM, Wide & Deep, xDeepFM
The 3 models use the Amazon Reviews dataset as a **regression** problem. The models are trained to predict the target rating of an item by a user and use square loss function and root mean square error (RMSE) and mean absolute error (MAE) as metrics.
### SLi-Rec, SASRec
The two models use the Amazon Reviews dataset as a **binary classification** problem. The models are trained to predict the probability of a user to review an item. The models are trained to reduce the binary cross entropy loss (logloss) and use area under the curve (AUC) as metric.<br>
- Because of the fundamental difference among the models, we cannot compare the performance of the models directly. The two sequential models (SLi-Rec & SASRec) already output two other metrics - **Normalized Discounted Cumulative Gain (NDCG)** and **Hit Rate**, we added code to output the same metric from the other 3 models for this study.
- To avoid heavy computation on all user-item pairs (a cross join of 5.6M users and 63K items!), the authors for the models followed a strategy based on positive and negative sampling. For each user <i>u</i> in the test dataset, we randomly sample 50 negative items, and rank these items with the ground truth item. Based on the rankings of these 51 items, NDCG@10 and Hit@10 is evaluated for all models.

### Comparing models based on NDCG@10, Hit@10

|     | Collaborative filtering |     | Content-based filtering | Hybrid |     |
| --- | ----------------------- | --- | ----------------------- | ------ | --- |
|  | SLi-Rec | SASRec | LightGBM | Wide & Deep | xDeepFM |
| NDCG@10 | **0.4128** | 0.3929 | 0.0725 | 0.1256 | 0.1881 |
| Hit@10 | **0.6699** | 0.61 | 0.1631 | 0.2781 | 0.3497 |

### Modeling Details
- LightGM
