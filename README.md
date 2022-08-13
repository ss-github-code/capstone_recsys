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
