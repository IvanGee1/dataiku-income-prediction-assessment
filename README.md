# dataiku-income-prediction-assessment
The following is my technical assessment submission repo for Dataiku March 2026 - Snr Data Scientist Position.  

# Objective
Predict whether an individual earns more than USD 50k using captured metrics from the census conducted

# Approach

1.Exploratory Data Analysis in the main data analysis notebook 

2.Preprocessing was handled in preprocessing.py

3.feature engineering implemented in feature_engineering.py

4.The final train.py file generates a complete report on the testing and findings. I explored using linear regression and random forest. These were both run against a baseline where no feature engineering was done to evaluate the improvement and effectiveness of my feature engineering. 


# Project Structure
Main files can be found in src folder, while the eda is the notebooks folder. To ensure a timely submission I did not store any trained models or pre processed data. 

# How to Run

Timply in terminal run "python  src/train.py"

The same can be done for the other 2 files as src/preprocessing.py and src/feature_engineering.py to see their outputs, but not required for code execution"

## Results
The follwoing in the output of running the train.py file :


### Logistic Regression - processed data

Number of input features: 36

Accuracy : 0.8559740286009021

Precision: 0.2888597640891219

Recall   : 0.8907209828645328

F1 Score : 0.4362455959779898

ROC-AUC  : 0.9460934663920179

Classification results:

              precision    recall  f1-score   support

           0     0.9915    0.8537    0.9174     92692

           1     0.2889    0.8907    0.4362      6186

    accuracy                         0.8560     98878

   macro avg     0.6402    0.8722    0.6768     98878

weighted avg     0.9476    0.8560    0.8873     98878

### Logistic Regression - processed and feature engineered

Number of input features: 43

Accuracy : 0.8607678148829871

Precision: 0.2963520120345995

Recall   : 0.8916909149692854

F1 Score : 0.4448566474454615

ROC-AUC  : 0.9484076046296173

Classification results:

              precision    recall  f1-score   support

           0     0.9917    0.8587    0.9204     92692
           1     0.2964    0.8917    0.4449      6186

    accuracy                         0.8608     98878

   macro avg     0.6440    0.8752    0.6826     98878

weighted avg     0.9482    0.8608    0.8907     98878

### Random Forest - processed data and feature engineered

Number of input features: 43

Accuracy : 0.931086793826736

Precision: 0.46805696846388606

Recall   : 0.7437762689945037

F1 Score : 0.5745504495504495

ROC-AUC  : 0.9498682623646602

Classification results:

              precision    recall  f1-score   support

           0     0.9822    0.9436    0.9625     92692
           1     0.4681    0.7438    0.5746      6186

    accuracy                         0.9311     98878

   macro avg     0.7251    0.8437    0.7685     98878

           0     0.9822    0.9436    0.9625     92692
           1     0.4681    0.7438    0.5746      6186

    accuracy                         0.9311     98878

   macro avg     0.7251    0.8437    0.7685     98878
           1     0.4681    0.7438    0.5746      6186

    accuracy                         0.9311     98878

   macro avg     0.7251    0.8437    0.7685     98878

    accuracy                         0.9311     98878

   macro avg     0.7251    0.8437    0.7685     98878

    accuracy                         0.9311     98878

   macro avg     0.7251    0.8437    0.7685     98878

weighted avg     0.9500    0.9311    0.9382     98878

### Feature engineering impact for Logistic Regression:

- Input features: 36 -> 43

- Accuracy change : 0.004793786282085044

- Precision change: 0.0074922479454775925

- Recall change   : 0.0009699321047526022

- F1 change       : 0.008611051467471709

- ROC-AUC change  : 0.0023141382375994057


### Model comparison on engineered features:

- Logistic Regression F1: 0.4448566474454615

Feature engineering impact for Logistic Regression:

- Input features: 36 -> 43

- Accuracy change : 0.004793786282085044

- Precision change: 0.0074922479454775925

- Recall change   : 0.0009699321047526022

- F1 change       : 0.008611051467471709

- ROC-AUC change  : 0.0023141382375994057

### Model comparison on engineered features:

- Logistic Regression F1: 0.4448566474454615

- Precision change: 0.0074922479454775925

- Recall change   : 0.0009699321047526022

- F1 change       : 0.008611051467471709

- ROC-AUC change  : 0.0023141382375994057

### Model comparison on engineered features:

- Logistic Regression F1: 0.4448566474454615

### Model comparison on engineered features:

- Logistic Regression F1: 0.4448566474454615

- Random Forest F1     : 0.5745504495504495

- Accuracy diff : 0.07031897894374883

- Precision diff: 0.17170495642928657

- Recall diff   : -0.14791464597478177

- F1 diff       : 0.12969380210498804

- ROC-AUC diff  : 0.0014606577350428918

### Top 10 Random Forest features:

- occupation code                    0.125392

- weeks worked in year               0.105645

- age                                0.088755

- num persons worked for employer    0.072744

- industry code                      0.055369

- worked_last_year                   0.052908

- sex                                0.046807

- dividends from stocks               0.040585

- education                          0.037968

- dividends log                      0.037053


The results above show us clearly that:

Random Forest is the preferred model due to better F1 score and balanced performance

Feature engineering added measurable value but did not drastically change model behaviour

The primary challenge in this problem is class imbalance, not model complexity. this is due to the high amount of census data where people were earning below 50k USD per year. 

The Top 10 features show us the attributes with the highest contribution to being able to predict which people will be earning above the 50k USD value.

# Potential improvements:

- Perform hyperparameter tuning (GridSearch / RandomSearch)
- Explore more advanced models such as Gradient Boosting

Introduce ML Ops considerations:

- Model versioning can be introduced to maintain a training history, including model parameters, feature sets, and performance metrics
- Monitoring for data drift and include tracking feature distributions against baselines
- Monitor key metrics like F1, precision and recall over time to alert when a retraining or intervention my be required
