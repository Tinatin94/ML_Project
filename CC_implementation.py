import numpy as np
import sklearn
import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import datetime

train_path = "/scratch/ab8690/ml/data/train.csv"
val_path = "/scratch/ab8690/ml/data/dev.csv"
save_path = "/scratch/ab8690/ml/"
MAX_ITER = 50

train = pd.read_csv(train_path, index_col=0)
val = pd.read_csv(val_path, index_col=0)

val = val[~val.labels.str.contains(":")]
train = train[~train.labels.str.contains(":")]

labels_list = [label.split(" ") for label in train['labels']]
labels_list = [label[0].split(",") for label in labels_list]
labels_list_val = [label.split(" ") for label in val['labels']]
labels_list_val = [label[0].split(",") for label in labels_list_val]
labels_list = [[int(s) for s in sublist] for sublist in labels_list] 
labels_list_val = [[int(s) for s in sublist] for sublist in labels_list_val] 


mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(labels_list)
encoded_labels_df = pd.DataFrame(encoded_labels, columns=mlb.classes_)
mlb = MultiLabelBinarizer()
encoded_labels_val = mlb.fit_transform(labels_list_val)
encoded_labels_df_val = pd.DataFrame(encoded_labels_val, columns=mlb.classes_)

def make_dict(entry):
    # entry is a list with form ['id:value', 'id:value']
    col_dict = {}
    for word in entry:
        key, value = word.split(":")
        key = int(key)
        value = float(value)
        col_dict[key] = value
    return col_dict
    
    
train_df = train['features']
val_df = val['features']

features = [item.split(" ") for item in train_df]
col_dicts = [make_dict(entry) for entry in features]

features_val = [item.split(" ") for item in val_df]
col_dicts_val = [make_dict(entry) for entry in features_val]

features_df = pd.DataFrame(col_dicts)
features_df_val = pd.DataFrame(col_dicts_val)

features_df = features_df.fillna(0)
features_df_val = features_df_val.fillna(0)
print('done cleaning')


###### DOWNSAMPLE #####
features_df = features_df.iloc[0:1000,:]
encoded_labels_df = encoded_labels_df.iloc[0:1000,:]
encoded_labels_df = encoded_labels_df.loc[:, (encoded_labels_df != 0).any(axis=0)]

X_train = np.array(features_df)
Y_train = np.array(encoded_labels_df)
x_val = np.array(features_df_val)
y_val = np.array(encoded_labels_df_val)


###### MODEL #######

start = datetime.datetime.now()
print ("Current date and time : ")
print (start.strftime("%Y-%m-%d %H:%M:%S"))


base_lr = LogisticRegression(max_iter = MAX_ITER, verbose=1, solver='liblinear')

int_rand = np.random.randint(1000)
chain = ClassifierChain(base_lr, random_state=int_rand)

chain.fit(X_train, Y_train)

filename = f"{MAX_ITER}_{int_rand}.sav"
file_path = save_path + filename

end = datetime.datetime.now()
print ("Current date and time : ")
print (end.strftime("%Y-%m-%d %H:%M:%S"))


###### SAVE MODEL PICKLE ######

pickle.dump(chain, open(file_path, 'wb'))

