import numpy as np
import sklearn
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import label_ranking_average_precision_score

import time


train = pd.read_csv("data/train.csv", index_col=0)
val = pd.read_csv("data/dev.csv", index_col=0)

train.shape, val.shape

# Remove broken label examples
def remove_broken(dataset):
    labels_list = [label.split(" ") for label in dataset['labels']]
    labels_list = [label[0].split(",") for label in labels_list]

    # drop the broken indices - found them using this - need to ask on Piazza what's wrong with them
    broken_indices = []
    for i in range(len(labels_list)):
        for j in range(len(labels_list[i])):
            try:
                int(labels_list[i][j])   
            except:
                #print(i, labels_list[i])
                broken_indices.append(i)

    labels_array = np.array(labels_list)
    print(len(labels_list))
    labels_list = np.delete(labels_array, broken_indices).tolist()
    print(len(labels_list))
    labels_list = [[int(s) for s in sublist] for sublist in labels_list] 
    return labels_list, broken_indices
    
labels_list, broken_indices = remove_broken(train)

val_labels_list, val_broken_indices = remove_broken(val)

def convert_labels(labels_list):
    mlb = MultiLabelBinarizer(classes = range(3993))
    encoded_labels = mlb.fit_transform(labels_list)
    encoded_labels_df = pd.DataFrame(encoded_labels, columns=mlb.classes_)
    return encoded_labels_df

encoded_labels_df = convert_labels(labels_list)

val_encoded_labels_df = convert_labels(val_labels_list)

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
    
def make_features_df(dataset,broken_indices):
    df = dataset['features']

    features = [item.split(" ") for item in df]
    col_dicts = [make_dict(entry) for entry in features]
    
    # Turn features column into sparse dataframe
    # Note: missing values as NaN - should these be zeros?
    features_df = pd.DataFrame(col_dicts)
    features_df.fillna(0)
    
    return features_df
    
features_df = make_features_df(train,broken_indices)

features_df = features_df.drop(broken_indices, axis=0)

val_features_df = make_features_df(val, val_broken_indices)

val_features_df = val_features_df.drop(val_broken_indices, axis=0)

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
svm = SVC(kernel='linear',
                 probability=True,
                 verbose=True,
                 max_iter=1,
                 decision_function_shape='ovr',
                 random_state=0)
model = OneVsRestClassifier(svm)

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
