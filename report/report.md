# Capstone Project Report (August 2022)

## Problem Statement
### Recommendations: What and Why
As explained in the advanced course on [Recommendation Systems](https://developers.google.com/machine-learning/recommendation/overview) by Google, ML-based recommendation model is responsible for determining the similarity of items in a large collection of items (movies, e-commerce, videos, etc.) and then coming up with a recommendation based on a user's likes or interests. The goal of a recommendation system is to find a small number `k` (maybe 10, 20) of compelling items from a large collection of millions of items. Yes, the user can initiate a search to access content. However, a recommendation system can display related items that users might not even have thought to search.

Two kinds of recommendations are commonly used:

- Recommending the **top `k`** items to a user on their homepage personalized for that user based on their known interests/history
- Recommending the **related set of `k`** items once a user has clicked on/viewed/purchased a product

One common pipeline for Recommendation Systems consists of the following components:
- Candidate Generation
- Scoring
- Ranking

#### Candidate Generation
The system starts with a huge corpora and builds and trains one or more models using item and user features provided as an input to the model. The exact nature of features depends on the type of model chosen.

There are two main approaches to candidate generation:
- **Collaborative Filtering**: Item and user similarity is used simultaneously to provide recommendations.
- **Content-based Filtering**: Item similarity is used to recommend items similar to what the user has liked before.

Both approaches map each **item** and each **user** (referred to as a **query** as we can also associate additional context to a user, e.g. item history) to an embedding vector in a common embedding space **E = R<sup>d</sup>**. Typically, the embedding space is low-dimensional where the dimension `d` is much smaller compared to the size of the corpus.<br>

For this project, we started with the simple collaborative filtering method: **matrix factorization**. In this model, given the rating matrix **A ∈ R<sup>mxn</sup>**, where **`m`** is the number of users (or queries) and **`n`** is the number of items, the model learns a user embedding matrix **U ∈ R<sup>m x d</sup>** and an item embedding matrix **V ∈ R<sup>n x d</sup>**. The embeddings are learned such that the product **UV<sup>T</sup>** is a good approximation of the rating matrix **A**. Note that the matrix **A** is very sparse - very few ratings are given by a user among the millions of items in the corpus.<br>
However, this simple model has one major drawback - it is enormously difficult to use side features of items e.g. genres or categories of the item, item history, etc. Also, popular items tend to be recommended for everyone without capturing specific user interests. We looked at alternatives to matrix factorization in order to address these limitations.

In recent years, a number of approaches based on gradient boosted decision trees and deep neural networks have been proposed. These approaches can easily take user and item features as input and can be trained to capture the specific interests of a user in order to improve the relevance of recommendations. We chose the following 5 models in this study for candidate generation:

| Collaborative Filtering  | Content-based Filtering | Hybrid   
| ------------------------ | ----------------------- | --------    |
| **S**hort-term and **L**ong-term<br>preference **I**ntegrated<br>**Rec**ommender (SLi-Rec) | LightGBM | **Wide & Deep** Learning<br>for Recommender<br>Systems |
| **S**elf-**A**ttentive **S**equential<br>**Rec**ommendation (SASRec) | | e**x**treme **Deep**<br>**F**actorization **M**achine (xDeepFM) |

#### Scoring
After training, the model is used to generate a score for each unseen item for a given user.
#### Ranking
The generated scores are sorted in the descending order and the top k items are provided as recommendations to the user. The top k items are across several categories. From that sorted list, we can also provide the top k items for a related category.<br>

In this project, **we had the following goals**:
- Review the published papers associated with each model (except the LightGBM based model).
- Learn how to train and deploy each model on our chosen dataset.
- Measure the overall performance of each model on our chosen dataset using the popular normalized discounted cumulative gain (NDCG@10) and hitrate (Hit@10) metrics.
- Compare and contrast the relevance of the top 10 recommended items by each model.

## Data Preparation: Amazon Reviews Dataset
[Amazon Reviews](http://deepyeti.ucsd.edu/jianmo/amazon/index.html) dataset has 157+ million reviews, and 15+ million items.

We took the following steps to reduce the size of data (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_gen_dataset.ipynb)):
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

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/item_dist.png?raw=true" target=”_blank” alt="Log count of items vs category"/>

- For LightGBM, Wide & Deep, and xDeepFM, we consider both the main category as well as the sub-categories shown above.
- For SLi-Rec and SASRec, we only consider the main category as the code for the models does not have support for sub-categories. The item distribution for the 12 main categories is shown below.
<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/item_dist_main.png?raw=true" target=”_blank” alt="Log count of items vs main category"/>

- The code for the data visualizations shown above and others presented in the report can be found here: (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/eda.ipynb)).

- **Chronological splitting** the Amazon reviews dataset into train, validation, and test datasets. While the sequential models (SLi-Rec, SASRec) have an elaborate strategy to split the data chronologically into train, validation, and test datasets (the last record in the chronological sequence of reviews goes to the test, the second last to the validation, and the remaining to the train), we had to deploy the same strategy for the 3 models. We used the `python_chrono_split` from the [Microsoft Recommenders](https://github.com/microsoft/recommenders) framework that includes stratification and is available [here](https://github.com/microsoft/recommenders/blob/main/recommenders/datasets/python_splitters.py).
<a id='ffm_format'></a>
- Besides the chronological split, xDeepFM requires the data to be in Field-aware Factorization Machine (FFM) format where each row in the dataset has the following format: `<label> <field_index_id>:<feature_index_id>:<feature_value>`. (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_ffm.ipynb))

## Modeling: Review of model architectures under study
The following 5 models were used for candidate generation, scoring and ranking:
### 1. LightGBM: A Highly Efficient Gradient Boosting Decision Tree

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_lightgbm.png?raw=true" alt="The architecture of LightGBM"/>

Notes about the model:<br>
- Model with fast training speed and high efficiency
- Support for parallel, distributed, and GPU learning
- We only require ordinal encoder to encode string like categorical features.

### 2. [Wide & Deep](https://arxiv.org/abs/1606.07792): Wide & Deep Learning for Recommender Systems
The following illustration of the architecture is taken from the [paper](https://arxiv.org/abs/1606.07792):

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_widendeep.png?raw=true" alt="The architecture of Wide & Deep"/>

Notes about the model:<br>
- Memorization and Generalization: The model **memorizes** the correlation available in data due the frequent co-occurrence of items or features. However, correlations are transitive; so we need to **generalize** in order to discover new feature combinations that have never or rarely occurred in the past.<br>
- **Wide** component is a linear model (y = w<sup>T</sup>x + b); here x are raw input features and **cross-product transformations**, w = weights, and b = bias.<br>
- **Deep** component is a feed-forward neural network; where the inputs are categorical features converted into embedding vectors. Each hidden layer performs the following computation where l = layer, f = activation function (ReLU), a(l), b(l), and w(l) are activations, bias, and weights at l-th layer: a<sup>(l+1)</sup> = f(w<sup>(l)</sup>a<sup>(l)</sup> + b<sup>(l)</sup>)<br>

### 3. [xDeepFM](https://arxiv.org/abs/1803.05170): Combining Explicit and Implicit Feature Interactions for Recommender Systems
The following illustration of the architecture is taken from the [paper](https://arxiv.org/abs/1803.05170):

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_xdeepfm.png?raw=true" alt="The architecture of Wide & Deep"/>

Notes about the model:<br>
- It shares some of the same attributes as the Wide & Deep model. However, unlike the previous model, xDeepFM does not require manual feature engineering for cross product transformations.
- **Compressed Interaction Network** (CIN) is used to generate cross feature interactions in an explicit manner. The details of CIN are quite similar to those of Convolutional Neural Networks (CNN). CIN lets the neural network learn cross feature interactions.
- The model combines CIN with a classical deep neural network (just like the **Deep** in Wide & Deep).
- Requires data according to the format required. Each row has a label (rating), and tab separated field_id:feature_id:feature_value for both numeric (user id, item id) and categorical (category/sub-category) features.<br>
<a id='sli_rec_arch'></a>
### 4. [SLi-Rec](https://www.microsoft.com/en-us/research/uploads/prod/2019/07/IJCAI19-ready_v1.pdf): Adaptive User Modeling with Long and Short-Term Preferences for Personalized Recommendation
<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_slirec.png?raw=true" alt="The architecture of SLi-Rec"/>

Notes about the model:<br>
- The model focuses on solving two key problems: **dynamic time intervals** and **dynamic latent intent**. Typical user interactions have dynamic time intervals, e.g. consider two actions 5 minutes apart vs. two actions days apart. This kind of temporal distance deserves special handling. The model uses an upgraded Long Short-Term Memory (LSTM) layer called Dynamic RNN as shown above.
- In addition, the user’s intent is also dynamic and changes from session to session. Irrelevant actions are useless for predicting a user's future action. E.g. suppose a user's review history is (iPhone-xs, airpods, cat food), when we want to recommend laptops (Macbook), only the first two actions make sense. To cope with content-aware distance, the authors use an attention mechanism (ATTN FCN) as shown above.
- In order to capture the static components influencing users' behaviors, which reflect their long-term behavior, the authors adopt the attentive "Asymmetric SVD" paradigm.
- Finally, the model uses an attention based fusion method to adapt to short and long term preferences:
  - **When**: if the next action occurs shortly after the last action; short-term information plays a major role in prediction; otherwise the long-term component weighs more.
  - **What**: if recent actions share a distinct intent/preference, then the next action may have a higher probability to share the same intent. (iPhones, airpods,.... MacBook?)<br>

### 5. [SASRec](https://arxiv.org/abs/1808.09781): Self-Attentive Sequential Recommendation

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/architect_sasrec.png?raw=true" alt="The architecture of SASRec"/>

Notes about the model:<br>
- Unlike existing sequential models that use convolutional or recurrent modules, SASRec uses a Transformer based on the 'self-attention' mechanism.
- This is the same mechanism used in Natural Language Processing that has proven to be highly effective in uncovering complex syntactic and semantic patterns between words in a sentence.
- Since the self-attention model does not include any recurrent or convolutional module, it is not aware of the positions of previous items. Hence, the authors inject a learnable position embedding layer.
- In order to tackle the problems of overfitting and vanishing gradients, the authors use both dropout and residual connections as shown above.

## Modeling: Training, validation, and test
We used the [Microsoft Recommenders](https://github.com/microsoft/recommenders) for this study. The repository offers implementations of the models under study and provides examples and best practices for building recommendation systems. However, we did make changes to the library's implementations when there were specific performance and process requirements (as discussed below).

### LightGBM, Wide & Deep, xDeepFM
The 3 models use the Amazon Reviews dataset as a **regression** problem. The models are trained to predict the target rating of an item by a user and use square loss function and root mean square error (RMSE) and mean absolute error (MAE) as metrics.
### SLi-Rec, SASRec
The two models use the Amazon Reviews dataset as a **binary classification** problem. The models are trained to predict the probability of a user to review an item. The models are trained to reduce the binary cross entropy loss (log loss) and use area under the curve (AUC) as metric.<br>
<a id='ndcg_10'></a>
- Because of the fundamental difference among the models, we cannot compare the performance of the models directly. The two sequential models (SLi-Rec & SASRec) already output two other metrics - **Normalized Discounted Cumulative Gain (NDCG)** and **Hit Rate**, we added code to output the same metric from the other 3 models for this study.
- To avoid heavy computation on all user-item pairs (a cross join of 830K+ users and 63K items!), the authors for the models followed a strategy based on positive and negative sampling. For each user <i>u</i> in the test dataset, we randomly sample 50 negative items, and rank these items with the ground truth item. Based on the rankings of these 51 items, NDCG@10 and Hit@10 are evaluated for all models.
<a id='main_results'></a>
### Comparing models based on NDCG@10, Hit@10

|     | Collaborative filtering |     | Content-based filtering | Hybrid |     |
| --- | ----------------------- | --- | ----------------------- | ------ | --- |
|  | SLi-Rec | SASRec | LightGBM | Wide & Deep | xDeepFM |
| NDCG@10 | **0.404** | 0.392 | 0.0725 | 0.1256 | 0.1881 |
| Hit@10 | **0.6654** | 0.628 | 0.1631 | 0.2781 | 0.3497 |

### Modeling: Implementation details
#### 1. LightGBM
In this model, categorical features were encoded using the ordinal encoder from the [Category Encoders](https://contrib.scikit-learn.org/category_encoders/) library. (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/modeling/lightgbm/lightgbm_amzn_electronics.ipynb))
- The validation loss and the 5 most important features from the feature importance list are shown below. Note that the model training is stopped due to early stopping as the validation rmse does not improve after 20 rounds. Here C3, C11, and C15 categories are "All Electronics", "Cell Phones & Accessories", and "Computers" categories/genres respectively.

| Validation Loss (RMSE) | 5 most important features |
| ---------------------- | ------------------------- |
| <img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/lightgbm_valid_loss.jpg?raw=true" alt="Validation loss LightGBM model"/> | <img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/lightgbm_feature_importance.jpg?raw=true" alt="5 most important features for the LightGBM model"/> |

- The model performance on the unseen test data: RMSE: 1.453, MAE: 0.901 was the worst when compared to Wide & Deep and xDeepFM models.
- We added code to enable the NDCG@10 and Hit@10 calculations based on all the users in the test set and using 50 negative samples for every positive sample as explained [here](#ndcg_10) taking care of the use of ordinal encoded categorical features.
<a id='top_k_user'></a>
- In addition, we added code to output the top k recommendations for a user. For this study, we chose a user who has the most reviews in our dataset. We had the model output predicted scores for all the items not reviewed by the user, sort the results in the descending order and save it in csv format. We summed up the results from this model and the next two regression models to come up with a top `k` list of recommendations to be displayed in a Streamlit [dashboard](https://ss-github-code-capstone-recsys-appstreamlit-rec-vxd02t.streamlitapp.com/).

#### 2. Wide & Deep
In this model, categorical features were encoded using the [`MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) from `sklearn.preprocessing` library. In addition, there are a series of preprocessing steps to convert the Amazon reviews dataset into the input format required by the model. These are shown in the Jupyter notebook [here](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_gen_input_wide_deep.ipynb). (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/modeling/wide_n_deep/wide_deep_electronics.ipynb))
- One important code change that we had to make in order to train the model using the Recommenders library on a GPU was as follows. The library's implementation of `pandas_input_fn` uses TensorFlow's `tf.data.Dataset.from_tensor_slices` api. This would try to load the entire dataframe onto the GPU and fail to do so. Instead we changed the function to use `tf.data.Dataset.from_generator` api. (source code: [Python](https://github.com/ss-github-code/capstone_recsys/blob/main/recommenders/utils/tf_utils_mod.py))
- The validation loss (RMSE and MAE) are shown below.
<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/wide_deep_valid_loss.jpg?raw=true" alt="Validation loss Wide & Deep model"/>

- The model performance on the unseen test data: RMSE: 1.13, MAE: 0.792 was lower than that of the LightGBM model.
- We added code to enable the NDCG@10 and Hit@10 calculations based on all the users in the test set and using 50 negative samples for every positive sample as explained [here](#ndcg_10) taking care of the encoding requirement of the categorical features.
- In addition, we added code to output the top k recommendations for a user as explained [here](#top_k_user).

### 3. xDeepFM
In this model, numerical and categorical features had to be input using the FFM format. The preprocessing step is discussed [here](#ffm_format). In addition, we had to perform hyperparameter tuning in order to prevent overfitting. (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/modeling/xdeepfm/xdeepfm_electronics.ipynb))
- The training and validation loss (RMSE) during the model (built with the tuned parameters: dropout rate: 0.5, L2 regularization: 0.01) training are shown below. The best model performance occurs at epoch 13 before the model still overfits and the training is stopped.
<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/xdeepfm_train_valid_loss.jpg?raw=true" alt="Validation loss xDeepFM model"/>

- The model performance on the unseen test data: RMSE: 1.1891 is comparable to that of the wide & deep model.
- We added code to enable the NDCG@10 and Hit@10 calculations based on all the users in the test set and using 50 negative samples for every positive sample as explained [here](#ndcg_10) taking care of the FFM format requirement for the numerical and categorical features.
- In addition, we added code to output the top k recommendations for a user as explained [here](#top_k_user).

### Summary for the regression based models
|     | Test RMSE |
| --- | --------- |
| LightGBM | 1.453 |
| Wide & Deep | **1.13** |
| xDeepFM | 1.189 |

### 4. SLi-Rec
The model requires the user, item and category vocabulary dictionaries mapping the alphanumeric userID, itemID and string categories to integers. In addition, the input to the model requires the data to be prepared in a series of steps that generate the 3 dictionaries (as pickle files) and the required train, validation and test datasets. The preprocessing steps to convert the Amazon review dataset are shown [here](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/amzn_gen_input_slirec.ipynb). 
- We set up the hyperparameters according to the authors' suggestions in the paper. Dimension for item/category embedding and RNN hidden layers is 18, while the dimension for the layers in [FCN](#sli_rec_arch) are set to 36. Learning rate is 0.001, L2 regularization is 0.0001 and no dropouts are used. (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/modeling/slirec/slirec_electronics.ipynb))
- The training and validation loss (log loss) during the model training are shown below. In addition, the rating metric AUC, and the pairwise ranking metric NDCG@6 plots for the validation data are shown below.

| Log Loss (Train, Validation) | Validation AUC, NDCG@6 |
| ---------------------------- | ---------------------- |
| <img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/sli_train_valid_logloss.png?raw=true" alt="Loss SLi-Rec model"/> | <img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/sli_valid_auc_ndcg6.png?raw=true" alt="Validation AUC and NDCG@6"/>

### 5. SASRec
The model requires the user and item vocabulary dictionaries mapping the alphanumeric userID and itemID to integers. The process of generating negative samples is handled by the `WarpSampler` object from the Recommenders library during training. However, the library's implementation would randomly pick users during the training. We changed the implementation to pick users after shuffling the set of users once at the beginning and when the sampler ran out of users to sample as shown here: (source code: [Python](https://github.com/ss-github-code/capstone_recsys/blob/main/recommenders/models/sasrec/train_sampler.py)). As with xDeepFM and SLi-Rec models, we performed hyperparameter tuning in order to prevent overfitting. (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/modeling/sasrec/sasrec_electronics.ipynb))
- Another addition we added was to enable batch processing of both validation and test sets. The implementation in the library would process only 10,000 randomly chosen users during the validation process and that too one user at a time. This would lead to extremely slow processing of validation and test datasets. We implemented code to enable batch processing of the entire validation and test datasets as shown here: (source code: [Python](https://github.com/ss-github-code/capstone_recsys/blob/main/recommenders/models/sasrec/valid_test_sampler.py)). This ensures that the entire validation (and test) dataset can be completed in about a minute and speeds up the hyperparameter tuning significantly.
- We also made minor changes to the implementation of the SASRec model to return the training and validation log loss as well as the pairwise metrics NDCG@10, Hit@10. We added code to save the model weights at the end of each training epoch as shown here: (source code: [Python](https://github.com/ss-github-code/capstone_recsys/blob/main/recommenders/models/sasrec/model.py)).
- Tuning the hyperparameters took longer than usual. We started with the default parameters (hidden units=100, dropout rate=0.5, L2 regularization=0) and observed overfitting (training loss decreased, while the validation loss increased; just to clarify this occurred even when using the default implementation of the training and validation loops and it did not improve even when we changed it to batch validation). We referred to the authors' implementation, and the paper. Even when setting the dropout rate to 0.95, we have not been able to satisfactorily prevent overfitting as shown below.
- The training and validation loss (log loss) during the model training are shown below. In addition, the pairwise ranking metrics NDCG@10 and Hit@10 plots for the validation data are shown below.

| Log Loss (Train, Validation) | Validation NDCG@10, Hit@10 |
| ---------------------------- | -------------------------- |
| <img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/sas_train_valid_logloss.png?raw=true" alt="Loss SASRec model"/> | <img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/sas_ndcg_hit10.png?raw=true" alt="NDCG@10 and Hit@10"/>

- Similar to the regression based models, we added output the top k recommendations for a user as explained below.

## Model Serving
After training each model, we save the best model along with the trained weights. The trained model can be used for serving the predicted rating score for a user reviewing an item using the regression models or the predicted probability score or logit for a user reviewing the item next using the binary classification models.

### Computing the Top K recommendations for a user
For each model, we have added code to compute the top k recommendations for any user in their respective notebooks. The steps for computing the top k recommendations are as follows:
- Identify the list of products that have not been reviewed by the user.
- For each such product not reviewed by the user, identify the features required by each model to predict either a rating score (LightGBM, Wide & Deep, xDeepFM) or a probability score or logit of that item being reviewed by the user next given the review history of the user (SLi-Rec, SASRec).
- Transform the input features according to the input requirements for each model.
- Use the model's predict function to output the score for each product. Sort the scores for each product and we have the Top K recommendations across all categories. The generated dataframe also has the category column and we can use that information to generate Top K recommendations for a specific category.

### Analyzing the review history of a sample user
- For this study in order to compare the quality of recommendations made, **we chose the user that had the most reviews** in our dataset.
- This user had 324 product reviews. The 10 most recent ones in the training dataset were as follows:

| Date | Main category | Other categories | Title |
| ---- | ------------- | ---------------- | ----- |
| 2018-03-25 | Home Audio & Theater | Electronics, Accessories & Supplies, Audio & Video Accessories | BlueRigger High Speed Micro HDMI to HDMI cable with Ethernet (10 Feet) - Support 4K- UltraHD, 3D, 1080p (Latest Standard) |
| 2018-03-25 | All Electronics | Electronics, Computers & Accessories, Computer Components | Corsair CMSA8GX3M2A1066C7 Apple 8 GB Dual Channel Kit DDR3 1066 (PC3 8500) 204-Pin DDR3 Laptop SO-DIMM Memory 1.5V |
| 2018-03-25 | All Electronics | Electronics, Computers & Accessories |	D-Link 8 Port 10/100 Unmanaged Metal Desktop Switch (DES-108) |
| 2018-03-25 | Computers | Electronics, Computers & Accessories |	StarTech.com CABSHELF Black Standard Universal Server Rack Cabinet Shelf |
| 2018-03-13 | All Electronics | Office Products, Office Electronics | HP Laserjet Pro M402dw Wireless Monochrome Printer, Amazon Dash Replenishment Ready (C5F95A#BGJ) |
| 2018-03-13 | All Electronics | Electronics, Accessories & Supplies, Audio & Video Accessories |	VCE 4K x 2K Mini HDMI Male to HDMI Female Converter Adapter Cable-6 Inch |
| 2018-03-13 | Computers | Electronics, Computers & Accessories, Computer Components | Timetec Hynix IC 4GB DDR3L 1600MHz PC3L-12800 Unbuffered Non-ECC 1.35V CL11 2Rx8 Dual Rank 204 Pin SODIMM Laptop Notebook Computer Memory Ram Upgrade (Dual Rank 4GB) |
| 2017-09-08 | Home Audio & Theater | Home & Kitchen | VIVO Universal LCD LED Flat Screen TV Table Top Desk Stand with Glass Base fits 32" to 55" T.V. (STAND-TV00L) |
| 2017-09-08 | All Electronics | Home & Kitchen | VIVO Universal LCD Flat Screen TV Table Top Stand / Base Mount fits 27" to 55" T.V. (STAND-TV00T) |
| 2017-08-17 | Home Audio & Theater | Electronics, Computers & Accessories, Computer Accessories & Peripherals | CyberPower  CP1500AVRLCD Intelligent LCD UPS System, 1500VA/900W, 12 Outlets, AVR, Mini-Tower |

- The validation record for this user is :

| Date | Main category | Other categories | Title |
| ---- | ------------- | ---------------- | ----- |
| 2018-03-25 | All Electronics | Electronics, Audio & Video Accessories | ESYNIC DAC Digital to Analog Audio Converter Optical Coax to Analog RCA Audio Adapter with Optical Cable 3.5mm Jack Output for HDTV Blu Ray DVD Sky HD Xbox 360 TV Box |

- And finally the test record for this user is :

| Date | Main category | Other categories | Title |
| ---- | ------------- | ---------------- | ----- |
| 2018-03-25 | Computers| Electronics, Computers & Accessories | New iPad 9.7" (2018 & 2017) / iPad Pro 9.7 / iPad Air 2 / iPad Air Screen Protector, SPARIN Tempered Glass Screen Protector - Apple Pencil Compatible/High Definition/Scratch Resistant |

- Note that **6 of the most recent actions of this user occur at the [same time](https://github.com/ss-github-code/capstone_recsys/blob/main/data_preparation/eda.ipynb)**. Now, you may ask: is that even possible? We will not dwell much on the issue of data quality, but this has important ramifications on the predicted output of SLi-Rec which has been trained to model both short-term and long-term preferences as discussed [earlier]().
- While the recent history would play a significant role in recommendations made by the sequential models (SLi-Rec, SASRec), we can also look at the histogram of the categories of products reviewed by this user overall (including sub categories used in LightGBM, Wide & Deep, xDeepFM) and just the main category alone (used in SLi-Rec, SASRec) as shown below (the average rating for each category is also shown to the right of every bar).

<a id='bar_plot_user'></a>
| Barplot of all categories in the product reviews by the user | Barplot of main categories in the product reviews by the user |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| <img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/hist_all_cat.png?raw=true" alt="Histogram of all categories in the reviews"/> | <img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/hist_main_cat.png?raw=true" alt="Histogram of main categories in the reviews"/> |

- Analyzing the information above (including the recent history), it is clear that the user has reviewed mostly "Computers" and "All Electronics" categories, though the most recent items include the "Home Audio & Theater" category. Also, note the **high average rating** given by the user for all categories of items.

### Top K recommendations for the sample user "explained"
- Unlike the use of Movielens dataset where the task of explaining (and comparing) the recommendations made by a model is a lot easier, the use of Amazon reviews dataset proved challenging. It is a lot easier to compare "Aladdin" to "Lion King" based on their genres than it is to compare computer parts, electronics, and accessories found in this dataset.
- Also, it is worth remembering that SLi-Rec, SASRec use collaborative filtering, LightGBM uses content-based filtering, and Wide & Deep, xDeepFM use both and hence come under hybrid model. On top of that, LightGBM, Wide & Deep, and xDeepFM models are solving regression problems while SLi-Rec, SASRec are solving binary classification problems. There were one too many knots to untangle! So we decided to compare the two problem types separately.

#### Comparing recommendations made by regression models
- In content-based filtering, we use item features to recommend other items similar to what the user likes, based on their previous ratings. As discussed [here](https://developers.google.com/machine-learning/recommendation/content-based/summary) the advantage of content-based filtering is that the model does not depend on any data from other users. The model can capture specific interests of a user and recommend niche items. So, how did we do among the 3 content-based models - LightGBM based, Wide & Deep, xDeepFM?<br>
All of them have similar inputs that include numerical and one hot encoded categorical features. The simple LightGBM does not consider any cross category transformations, the Wide & Deep does consider them, and the xDeepFM has a sophisticated neural net (CIN) to discover the cross category transformations. Of course, Wide & Deep and xDeepFM are hybrid models and also use collaborative filtering.<br>
- As already noted [above](#main_results), the performance of the models using the NDCG@10 and Hit@10 metrics does improve as we go from LightGBM model to Wide & Deep to xDeepFM. In a very sparse dataset, such as ours, it definitely helps to throw in additional features to improve performance for these metrics. But do the additional features and model complexity help with the recommendations made for this specific user? Instead of worrying about product titles (they are shown below truncated just for context), let's look at the main categories in the top 10 recommendations for this user.

| Rank | LightGBM Category | Title | Wide & Deep Category | Title | xDeepFM Category | Title |
| ---- | ----------------- | ----- | -------------------- | ----- | ---------------- | ----- |
| 1 | Amazon Devices, Cell Phones & Accessories, Accessories | Amazon Kindle 2 (2nd Generation) USB Car Charger... | Camera & Photo, Electronics, Computers & Accessories, Computer Accessories & Peripherals | WT-AF-5v10w 802.3af PoE Splitter... | Industrial & Scientific, Automotive | Scott 75143 Scott Shop Towels... |
| 2 | Camera & Photo, Cell Phones & Accessories, Accessories | Zeikos ZE-SG26R 3 Pieces Ultra... | Cell Phones & Accessories	| APPLE IPHONE 4... | Cell Phones & Accessories | Masha and the Bear... |
| 3 | Home Audio & Theater, Cell Phones & Accessories, Accessories | ANSMANN Individual cell battery... | Camera & Photo, Electronics	 | Pelican 1400 Case... | Industrial & Scientific, Automotive | FW1 Cleaner With... |
| 4 | Cell Phones & Accessories, Sports & Outdoors, Sports & Fitness, Accessories | CAVN 2-Pack Compatible... | Cell Phones & Accessories, Accessories	 | Galaxy S9 Plus Screen Protector... | Industrial & Scientific, Electronics, Accessories & Supplies | DROK Reusable 30 Pcs... |
| 5 | Cell Phones & Accessories, Sports & Outdoors, Sports & Fitness, Accessories | Getwow 10-Pack Silicon... | Cell Phones & Accessories, Electronics, Computers & Accessories, Computer Accessories & Peripherals | iPhone Charger... | Industrial & Scientific, Office Products, Office & School Supplies | E-Z Grader Grading... |
| 6 | Cell Phones & Accessories, Sports & Outdoors, Sports & Fitness, Accessories | Bike Phone Mount... | Computers, Electronics, Computers & Accessories, Computer Accessories & Peripherals | Monoprice 9 PIN... | Cell Phones & Accessories | APPLE IPHONE 4... |
| 7 | Cell Phones & Accessories, Sports & Outdoors, Sports & Fitness, Accessories | HanTop Apple Watch Band... | Industrial & Scientific, Tools & Home Improvement | 3M Half Facepiece... | Industrial & Scientific, Automotive | Breeze Power-Seal... |
| 8 | Cell Phones & Accessories, Sports & Outdoors, Sports & Fitness, Accessories | bayite Accessory... | Amazon Devices, Electronics, Computers & Accessories | Timbuk2 Nylon Quilted... | Industrial & Scientific, Office Products | IKEA BEKVAM... |
| 9 | Cell Phones & Accessories, Sports & Outdoors, Sports & Fitness, Accessories | TUSITA Fitbit... | Cell Phones & Accessories | Nillkin Premium Matte Hard... | Industrial & Scientific, Tools & Home Improvement | Apache 990... |
| 10 | Cell Phones & Accessories, Sports & Outdoors, Sports & Fitness, Accessories | Garmin vívokeeper... | Industrial & Scientific | Ultrasonic Cleaner ... | Camera & Photo, Electronics, Accessories | MagicFiber Microfiber... |

- At first glance, we cannot explain the predictions made by the 3 models. However, note that this user has been generous with their ratings giving high ratings to all categories of items as shown in the [barplots](#bar_plot_user). Armed with that knowledge and knowing that all 3 models are **regression** models, it became easier to "explain" the top 10 predictions made by the 3 models.
- In summary, it is easier to see why content-based filtering done by LightGBM seems to latch on to "Cell Phones & Accessories", and "Accessories" categories (all 10 recommendations are from that category!). It is much harder to explain the recommendations made by the hybrid models - Wide & Deep and xDeepFM where the item features go through a deep neural network and discover feature combinations that are not easily explained.

#### Comparing recommendations made by binary classification models (also collaborative filtering models)
- In collaborative filtering, we use similarities between users and items simultaneously to provide recommendations. As discussed [here](https://developers.google.com/machine-learning/recommendation/collaborative/summary), an advantage of collaborative filtering is that it can help user discover new interests (**serendipity**). The model might recommend an item simply because similar users are interested in that item.
- In addition, both models are sequential models - they are both trying to capture the sequential dynamics of the user's actions. While SLi-Rec has a sophisticated mechanism to capture both the short-term and the long-term preferences, SASRec is modeling the item sequence using the self-attention mechanism of a transformer.
- Let's take a look at what the two models' top 10 predictions were for this user. Once again, we can focus on the item categories instead of the item titles (they are shown truncated for brevity).

| Rank | SLi-Rec Category | Title | SASRec Category | Title |
| ---- | ---------------- | ----- | --------------- | ----- |
| 1 | Computers | ICY DOCK Dual... | Computers | NETGEAR N300 WiFi...|
| 2 | Computers | Ubiquiti Unifi... | Cell Phones & Accessories | OontZ Angle 3 |
| 3 | Computers | TP-Link AV1200... | All Electronics | AmazonBasics 6-Sheet Cross-Cut... |
| 4 | Computers | OWC In-Line Digital... | Cell Phones & Accessories | Anker PowerCore+ Mini... |
| 5 | Computers | Linksys Business... | Computers | SanDisk Ultra 32GB... |
| 6 | Computers | Crucial 32GB Kit... | All Electronics | SanDisk 2GB Class... |
| 7 | Computers | WT-GPOE-4-48v48w Gigabit... | Computers | Sabrent 4-Port USB 3.0... |
| 8 | Computers | CAT 6 Ethernet Cable... | Amazon Devices | Fire HD 8 Tablet with Alexa... |
| 9 | Computers | Ubiquiti NanoStation... | Amazon Devices | Fire HD 10 Tablet with Alexa... |
| 10 | Computers | UGREEN 2 Pack USB... | Amazon Devices | Fintie Slimshell Case... |

- As far as the predictions made by SLi-Rec model goes, based on the model's architecture and the fact that 6 of the last 10 reviews made by the user were made on the same day, it is not surprising that the top 10 recommendations of the SLi-Rec model all belong to the "Computers" category. As explored [previously](#bar_plot_user), the highest fraction of the user's reviews were in the "Computers" category. And it is not surprising that collaborative filtering based model will latch on to the "Computers" category as almost recent reviews made by the user fall under the umbrella of computer accessories and would normally be associated together.
- Please do note that we could not prevent overfitting while training the SASRec model. So, we will have to take that into consideration as we look for explanations for the behavior of the SASRec model. The model does seem to discover a few items that may be called serendipitous, but we do not want to draw too many conclusions on predictions made by an overfitting model.

### Top 10 recommendations in a [Streamlit dashboard](https://ss-github-code-capstone-recsys-appstreamlit-rec-vxd02t.streamlitapp.com/)
- The output from each of the 5 models was further processed to prepare for showing the top 10 recommendations under each category of item. (source code: [Jupyter Notebook](https://github.com/ss-github-code/capstone_recsys/blob/main/app/process_output.ipynb))
- Instead of displaying the results from the 5 models separately (it would be overwhelming to the viewer, especially since the items themselves are not as exciting as movies), we decided to do an ensemble where we averaged the scores for all regression models (LightGBM, Wide & Deep, and xDeepFM) and similarly average the scores of the binary classification models (SLi-Rec, SASRec).
- The dashboard will allow us to show the top `k` results in a much finer granularity and let the user explore the recommendations made in each category. (source code: [Python](https://github.com/ss-github-code/capstone_recsys/blob/main/app/streamlit_rec.py))
- The dashboard running as a Streamlit app can be accessed [here](https://ss-github-code-capstone-recsys-appstreamlit-rec-vxd02t.streamlitapp.com/).

<img src="https://github.com/ss-github-code/capstone_recsys/blob/main/report/images/topk_dash.png?raw=true" width="400" alt="Top K Dashboard"/>

## Conclusion

In this project, we reviewed published papers on [Wide & Deep](https://arxiv.org/abs/1606.07792), [xDeepFM](https://arxiv.org/abs/1803.05170), [SLi-Rec](https://www.microsoft.com/en-us/research/uploads/prod/2019/07/IJCAI19-ready_v1.pdf), and [SASRec](https://arxiv.org/abs/1808.09781) models that each offer a unique methodology to be used in a recommendation system. We coded the steps required to explore and prepare the Amazon dataset of reviews of items that fell under the broad category of "Electronics". Each model required a different input format and we had to transform the dataset into the format demanded by the model. We learnt how to train and deploy the models on the chosen reviews dataset. We measured the performance of the models using NDCG@10 and Hit@10 metrics. Finally, we were able to compare and contrast the relevance of the top 10 recommended items for the user with the most reviews in the dataset.<br>

We had to overcome a few challenges - the most difficult one was our inability to prevent overfitting while training the SASRec model. An enormous challenge that we had to overcome was in dealing with TensorFlow(TF) 1.x code under the hood in the Microsoft Recommenders library. When we made our initial project proposal, we had not factored the time to be spent in overcoming the challenges associated with learning and understanding TF 1.x code (and the lack of eager execution and print in TF 1.x). We partially overcame that hurdle and learnt how to port from TF 1.x to TF 2.x and were able to port and test TF 2.x implementation of xDeepFM (source code: [Python](https://github.com/ss-github-code/capstone_recsys/blob/main/recommenders/models/deeprec/models/xDeepFM_v2.py)). Since this was not the primary focus for this project, we did not pursue this further due to the paucity of time.<br>
There were quite a few data quality issues including missing values in the price column that made us drop our initial goal of finding recommendations in a given price range. This was to be expected in a dataset of this size built by scraping reviews and information from a live e-commerce site. Finally, we found it difficult to explain some of the recommendations made by some of the models and could only come up with plausible explanations instead of explanations based on solid facts.<br>

We overcame a couple of challenges associated with the underlying implementation in the open source library: (a) we were successful in using the TensorFlow generator mechanism when the underlying library was failing to load the whole training dataset in the GPU memory; (b) we sped up validation and testing by enabling batch validation and test for a model that only had predictions working on 1 record at a time and was super slow.<br>
 
Our main contribution in this project was to build a data pipeline that was used to train, validate and deploy 5 recommendation system models on a common dataset. The 5 models had widely different architectures and complexities and we were able to successfully learn and implement the data pipeline, perform hyperparameter tuning, and model serving for each of the 5 models (after overcoming the challenges mentioned above). Our end-to-end pipeline (including the interactive dashboard) will allow data scientists to explore other recommendation system models that are supported in the open source Recommenders library.

In the future, we would like to explore how to tune the hyperparameters of the SASRec model. It is worth noting that the number of hyperparameters in SLi-Rec and SASRec are quite large and it requires an enormous amount of compute resources to do a grid search over all of them. We would also like to consider the much larger dataset consisting of over 38 million+ reviews and see how well the models perform. We now understand how to use multi-processes and queues to feed input data to the models and would like to explore ways to speed up the xDeepFM, SLiRec models in order to cut down the time it takes to train these models. This would be very helpful when we train the models on the much larger dataset. We would also like to explore how to build an ensemble of models or combine the results from multiple models while serving top k recommendations for a user. Finally, we would like to learn how to use distributed compute resources to handle much larger datasets.

## References
- Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, Hemal Shah. Wide & Deep Learning for Recommender Systems. arXiv:1606.07792 [cs.LG], 2016.
- Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., & Sun, G. (2018). xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, KDD 2018, London, UK, August 19-23, 2018.
- Zeping Yu, Jianxun Lian, Ahmad Mahmoody, Gongshen Liu, Xing Xie. Adaptive User Modeling with Long and Short-Term Preferences for Personalized Recommendation. In Proceedings of the 28th International Joint Conferences on Artificial Intelligence, IJCAI’19, Pages 4213-4219. AAAI Press, 2019.
- Wang-Cheng Kang, Julian McAuley (2018). Self-Attentive Sequential Recommendation. In Proceedings of IEEE International Conference on Data Mining (ICDM'18)

## Statement of Work
All of the team members were involved in the initial exploration of the datasets and the consideration of the recommendation system models to pursue in this study. Shiv and Zhipeng looked at several different Amazon reviews dataset categories (Video Games, Fashion, etc.) before choosing the Electronics dataset. Shiv and Reitesh also looked at the [Dressipi dataset](http://www.recsyschallenge.com/2022/) that was part of the ACM Conference Series on Recommender Systems 2022. We were successful in implementing the SLi-Rec model for this challenge. Shiv and Zhipeng worked on understanding matrix factorization, and the regression models. Shiv and Reitesh worked on understanding the sequential models. Shiv looked at the TensorFlow API and made the changes required to train all of the models. He worked on tuning the hyperparameters, added code to compute the NDCG@10, Hit@10 metrics, and output the top 10 recommendations for each model. Shiv and Zhipeng worked on the dashboard using Streamlit.
