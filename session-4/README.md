# Session 4

Implement a Deep Learning App. Train a model to classify text reviews between positive and negative and deploy a web app to predict on user reviews. 
Then, deploy the model using Google Cloud App Engine.

## Dataset

We will use the Yelp Review polarity dataset ([link](https://www.kaggle.com/datasets/irustandi/yelp-review-polarity), [direct download](https://www.kaggle.com/api/v1/datasets/download/irustandi/yelp-review-polarity)).

The dataset has two labels. "1" corresponds to negative reviews, and "2" to positive reviews.

## Installation
### With Conda
Create a conda environment by running
```
conda create --name aidl-session4 python=3.10
```
Then, activate the environment
```
conda activate aidl-session4
```
and install the dependencies
```
pip install -r requirements.txt
```

## Running the project
To train the model run:
```
python session-4/train.py
```

To run the app run:
```
python session-4/app/main.py
```
then, you can test you app by going to http://localhost:8080