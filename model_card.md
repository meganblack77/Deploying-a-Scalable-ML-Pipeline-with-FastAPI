# Model Card

## Model Details

This model is a Logistic Regression classifier designed to predict whether an individual's income exceeds $50,000 based on various demographic features, such as education, occupation, and marital status.
It was trained on a Census Income dataset. 
The model was built for educational purposes, primarily to demonstrate the process of constructing a machine learning pipeline. 
It uses one-hot encoding for categorical features and standardization for continuous variables.


## Intended Use

It serves as a baseline model for predicting income levels and can be adapted for similar binary classification tasks.


## Training Data

The model was trained on a Census Income dataset, which consists of 32,561 samples after an 80/20 train-test split. 
The dataset includes demographic information such as workclass, education, marital status, occupation, relationship, race, sex, and native country. 
Categorical features were one-hot encoded, and the labels were binarized to indicate whether an individual's income exceeds $50,000.


## Evaluation Data

The model was evaluated on a test dataset derived from the same Census dataset, consisting of 8,141 samples. 
The test data underwent the same preprocessing steps as the training data, using the encoder and label binarizer fitted on the training set. 
The evaluation aimed to assess the model's generalization performance across various demographic slices.


## Metrics

The model achieved an overall precision of 0.7281, a recall of 0.2693, and an F1 score of 0.3931. 
These metrics indicate that while the model is reasonably precise in its predictions, it has a relatively low recall, meaning it misses a significant number of true positives. 
The model's performance was also evaluated on specific slices of the data, revealing varying results. 
For instance, the model performed perfectly on the "Workclass: Without-pay" slice (Precision: 1.0000, Recall: 1.0000, F1: 1.0000), but this result is likely influenced by the small sample size (Count: 4). 
In contrast, for the "Race: White" slice, with a much larger sample size (Count: 5,595), the model achieved a precision of 0.7327 and an F1 score of 0.3912, reflecting more realistic performance.


## Ethical Considerations

This model may reflect biases inherent in the training data, particularly regarding demographic features such as race and gender. 
These biases can lead to skewed predictions that may affect certain groups more than others.


## Caveats and Recommendations

This model may not generalize well to different populations or datasets with varying distributions. 
The model's performance varies significantly across different slices of the data, so this should be considered when interpreting results when testing with different datasets.
