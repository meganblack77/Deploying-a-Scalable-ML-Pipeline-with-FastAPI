import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
	)


X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)  
model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(encoder, encoder_path)
lb_path = os.path.join(model_dir, "label_binarizer.pkl")
save_model(lb, lb_path)

# load the model
model = load_model(model_path) 

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        # Iterate through the unique values in one categorical feature
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            # Save the performance on slices to a file
        print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
        print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
