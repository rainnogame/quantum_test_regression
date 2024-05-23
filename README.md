## Goal

You have a dataset (train.csv) that contains 53 anonymized features and a target column. Your task is to build a model
that predicts a target based on the proposed features. Please provide predictions for the hidden_test.csv file. Target
metric is RMSE. The main goal is to provide github repository that contains:

- jupyter notebook with exploratory data analysis;
- train.py python script for model training;
- predict.py python script for model inference on test data;
- file with prediction results;
- readme file that contains instructions about project setup and general guidance
  around project;
- requirements.txt file.

## Initial EDA

The EDA results are presented in the jupyter notebook `eda.ipynb`

EDA results:

- The dataset has 53 features and 1 target column
- There are no missing values in the dataset
- There are several groups of features
    - #0-5 and #9-12 - uniformly distributed positive discrete random variables
    - #13-52 - uniformly distributed discrete random variables
    - #6 - linearly distributes continuous symmetric random variable
    - #7 - linearly distributes continuous random variable limited by the range 0-1
    - #8 - categorical binary variable (0, 1). The variable is a step function of #6
    - target - linearly distributes continuous random variable wih uniform distribution
- From the EDA it's clear that the all the features except #6, #7 and #8 are not dependent on target
- #8 can be replaced by #6
- We have a quadratic dependency between #6 and target
- We have a linear dependency between #7 and target (small correlation)
- From the EDA it's clear that the best model for this dataset is a linear quadratic model in form `y=#6^2+#7`
- So we have two equal options for a model:
    - Linear regression with polynomial feature #6
    - Just a function that will exactly calculate the target value

## EDA Important notes

- I'm aware that everything can be done just brutforcing popular models and hyperparameters
    - E.G. trying Polynomial regression model or Tree-based model I will know from the start important features and get
      the results faster
- I selected the way I analyzed the data intentionally to show that you can solve a lot of tasks without any models
  training, just carefully analyzing the data and it's relations
- From the EDA it looks like I used some king of reverse engineering of the data to get the model. Actually because I
  know the contex of the task I expected that just linear regression will not be interesting and I will have some tricky
  data with non-informative or correlated features or some non-linear dependencies. This expectation was the main reason
  why I decided to concentrate on correlation analysis
- I think despite the fact that eventually I used the simplest model and actually can even not thain any model this
  approach is still legit

## Implementation notes

Implemented under `python v3.12`

To install requirements:

```bash
pip install -r requirements.txt
```

To train the model

```
usage: train.py [-h] [--data_path DATA_PATH] [--model_save_path MODEL_SAVE_PATH]

Train regression model

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the file with a data. Default: train.csv
  --model_save_path MODEL_SAVE_PATH
                        Path to save the model. Default: model.pkl
```

Example `python train.py --data_path data/train.csv --model_save_path data/model.pkl`

After training the model you can run prediction

```
usage: predict.py [-h] [--data_path DATA_PATH] [--model_path MODEL_PATH]
                  [--results_path RESULTS_PATH]

Train regression model

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the file with a data. Default: hidden_test.csv
  --model_path MODEL_PATH
                        Path to the trained model. Default: model.pkl
  --results_path RESULTS_PATH
                        Path to save the results. Default: results.csv

```

Example `python predict.py --data_path data/hidden_test.csv --model_path data/model.pkl --results_path data/results.csv`

## Implementation important notes

- To test the results I wrote a simple test (cannot be considered as production-ready unit-test) that can be run
  using `pytest`
- Some important code practices are not used because of the simplicity and context of the task
    - I didn't use any logging
    - I didn't use any configuration files
    - I didn't use any data validation
    - I didn't add any tests except the simplest one
- I didn't try to write the best code in terms of code quality
    - E.G. I used pandas for data loading, etc
- I loaded a big file to the repo just to keep it simple
