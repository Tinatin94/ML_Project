import numpy as np
import sklearn
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import label_ranking_average_precision_score

import time

train_path = "/scratch/ab8690/ml/data/train.csv"
val_path = "/scratch/ab8690/ml/data/dev.csv"

train = train[~train.labels.str.contains(":")]
val = val[~val.labels.str.contains(":")]

labels_list = [label.split(" ") for label in train['labels']]
labels_list = [label[0].split(",") for label in labels_list]
labels_list_val = [label.split(" ") for label in val['labels']]
labels_list_val = [label[0].split(",") for label in labels_list_val]

def convert_labels(labels_list):
    mlb = MultiLabelBinarizer(classes = range(3993))
    encoded_labels = mlb.fit_transform(labels_list)
    encoded_labels_df = pd.DataFrame(encoded_labels, columns=mlb.classes_)
    return encoded_labels_df

encoded_labels_df = convert_labels(labels_list)

val_encoded_labels_df = convert_labels(labels_list_val)

# Convert features

def make_dict(entry):
    # entry is a list with form ['id:value', 'id:value']
    col_dict = {}
    for word in entry:
        key, value = word.split(":")
        key = int(key)
        value = float(value)
        col_dict[key] = value
    return col_dict
    
def make_features_df(dataset):
    df = dataset['features']

    features = [item.split(" ") for item in df]
    col_dicts = [make_dict(entry) for entry in features]
    
    features_df = pd.DataFrame(col_dicts)
    features_df.fillna(0)
    
    return features_df
    
features_df = make_features_df(train)

val_features_df = make_features_df(val)


# Train and evaluate

# Train data
X = np.array(features_df).astype(float)
# Note: converted NaN to zeros
X[np.isnan(X)]=0

y = np.array(encoded_labels_df).astype(float)

# Val data
X_val = np.array(val_features_df).astype(float)
# Note: converted NaN to zeros
X_val[np.isnan(X_val)]=0

y_val = np.array(val_encoded_labels_df).astype(float)

# Define model
linsvm = LinearSVC(loss='hinge',
                       multi_class='ovr',
                       verbose=True)
model = OneVsRestClassifier(linsvm)

start = time.process_time()
model.fit(X,y)
elapsed_fit = time.process_time() - start

print("Time to fit model (min):",elapsed_fit/60)

start_predict = time.process_time()
y_pred = model.decision_function(X_val)
elapsed_predict = time.process_time() - start_predict

print("Time to predict (min):",elapsed_predict/60)

# Evaluate
y_true = y_val
LRAP = label_ranking_average_precision_score(y_true,y_pred)

print("LRAP:", LRAP)
