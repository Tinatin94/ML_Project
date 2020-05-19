import numpy as np
import sklearn
import pandas as pd

from sklearn.metrics import label_ranking_average_precision_score
from sklearn.neural_network import MLPClassifier

train_path = "/scratch/ab8690/ml/data/train.csv"
val_path = "/scratch/ab8690/ml/data/dev.csv"


train = pd.read_csv(train_path, index_col=0)
val = pd.read_csv(val_path, index_col=0)


val = val[~val.labels.str.contains(":")]
train = train[~train.labels.str.contains(":")]

print("train shape:", train.shape)
print("val shape:", val.shape)

labels_list = [label.split(" ") for label in train['labels']]
labels_list = [label[0].split(",") for label in labels_list]
labels_list_val = [label.split(" ") for label in val['labels']]
labels_list_val = [label[0].split(",") for label in labels_list_val]

print("labels len (15511):", len(labels_list))
print("val labels len (1314):", len(labels_list_val))

labels_list = [[int(s) for s in sublist] for sublist in labels_list]


labels_list_val = [[int(s) for s in sublist] for sublist in labels_list_val]

from sklearn.preprocessing import MultiLabelBinarizer
class_list = [i for i in range(0,3993)]
mlb = MultiLabelBinarizer(classes=class_list)
encoded_labels = mlb.fit_transform(labels_list)

encoded_labels_val = mlb.fit_transform(labels_list_val)
encoded_labels_df_val = pd.DataFrame(encoded_labels_val, columns = class_list)
encoded_labels_df = pd.DataFrame(encoded_labels, columns = class_list)


def make_dict(entry):
    # entry is a list with form ['id:value', 'id:value']
    col_dict = {}
    for word in entry:
        key, value = word.split(":")
        key = int(key)
        value = float(value)
        col_dict[key] = value
    return col_dict


# drop broken indices
train_df = train['features']
#train_df = train_df.drop(broken_indices, axis=0)
val_df = val['features']

features = [item.split(" ") for item in train_df]
col_dicts = [make_dict(entry) for entry in features]

features_val = [item.split(" ") for item in val_df]
col_dicts_val = [make_dict(entry) for entry in features_val]


# Turn features column into sparse dataframe
# Note: missing values as NaN - should these be zeros?
features_df = pd.DataFrame(col_dicts)
features_df_val = pd.DataFrame(col_dicts_val)


features_df = features_df.fillna(0)
features_df_val = features_df_val.fillna(0)


X_train = np.array(features_df)
Y_train = np.array(encoded_labels_df)
x_val = np.array(features_df_val)
y_val = np.array(encoded_labels_df_val)

print("X_train shape (15511, 5000):", X_train.shape

####### MLP Classifier #########

layers=[(100, 100),(100, 100, 100),(500, 500),(500, 500, 500),(1000,),(1000, 1000)]
lr =[0.001,0.0001]


results=[]
for i in layers:
    for j in lr:
        mlp = MLPClassifier()
        mlp.fit(X_train,Y_train)
        y_score = mlp.predict_proba(x_val)
        precision =label_ranking_average_precision_score(y_val, y_score)
        print(precision,i,j)
        results.append(precision)