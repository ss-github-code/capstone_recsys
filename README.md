# Capstone Project (July, August 2022)
Recommender Systems - A Deep Dive

## Problem Statement
[Recommendations: What and Why](https://developers.google.com/machine-learning/recommendation/overview)
- Recommending the top k items to a user on their homepage
- Recommending the related set of k items once a user has clicked on a product

One common pipeline for Recommendation Systems consists of the following components:
- Candidate Generation
- Scoring
- Ranking

### Candidate Generation
The system starts with a huge corpora and builds and trains a model in order to output a score to an item based on features provided as an input to the model. The exact nature of features depends on the type of model chosen.

There are two main approaches to candidate generation:
- Collaborative Filtering: Item and user similarity is used simultaneously to provide recommendations.
- Content-based Filtering: Item similarity is used to recommend items similar to what the user has liked before.

#### List of models considered in this study for candidate generation
| Collaborative Filtering  | Content-based Filtering | Hybrid   
| ------------------------ | ----------------------- | --------    |
| Short-term and Long-term<br>preference Integrated<br>Recommender (SLi-Rec) | LightGBM | Wide & Deep Learning<br>for Recommender<br>Systems |
| Self-Attentive<br>Sequential<br>Recommendation (SASRec) | | extreme Deep<br>Factorization Machine (xDeepFM) |

### Scoring
After training, the model is used to generate a score for each unseen item for a given user.
### Ranking
The generated scores are sorted in the descending order and the top k items are provided as recommendations to the user. The top k items are across several categories. We can also provide the top k items for a related category.

## Model Architecture
- [Wide & Deep](https://arxiv.org/abs/1606.07792): Wide & Deep Learning for Recommender Systems

![The architecture of Wide & Deep](https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_widendeep.png?raw=true)

- [xDeepFM](https://arxiv.org/abs/1803.05170): Combining Explicit and Implicit Feature Interactions for Recommender Systems

![The architecture of xDeepFM](https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_xdeepfm.png?raw=true)

- [SLi-Rec](https://www.microsoft.com/en-us/research/uploads/prod/2019/07/IJCAI19-ready_v1.pdf): Adaptive User Modeling with Long and Short-Term Preferences for Personalized Recommendation

![The architecture of SLi-Rec](https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_slirec.png?raw=true)
