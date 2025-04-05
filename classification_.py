import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.stats import pearsonr
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import f_regression
from scipy.stats import linregress
from scipy.stats import spearmanr
import csv 
from sklearn.preprocessing import Normalizer, StandardScaler

csv_input = pd.read_csv('features_final_no_diag.csv')
mask_dir = os.listdir('Original_Annotations/Original_Annotations/px1Label_SHT')
sig_csv = pd.read_csv("significance_prostatex.csv")
dir = os.listdir("Original_Sequences/Original_Sequences")
labels= []
data = []
pat_name = []
pat = {}
for index, row in sig_csv.iterrows():
        column= 'ProstateX1_' + str(row['PatientID']).zfill(4)
        if column in mask_dir:
             pat[column] = row['Sig']
labels = list(pat.values())
for index, row in csv_input.iterrows():
    if row['PatientID'] in dir:
            data.append(row)
selected_features = []
data = pd.DataFrame(data)
labels = pd.DataFrame(labels)

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

drop_x = X_train
drop_x.drop(columns=['PatientID'], axis=1, inplace=True)
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(drop_x), columns=drop_x.columns)

# drop_x = X_train
# drop_x.drop(columns=['PatientID'], axis=1, inplace=True)
# drop_x.iloc[:,:] = Normalizer(norm='l2').fit_transform(drop_x)
# print(drop_x, y_train)

selected_features = []
y_train_flat = y_train.values.ravel() 
with open('values3.csv', 'w', newline='') as csvfile:
        fieldnames = ['Features', 'P values', 'R values']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for feature in df_normalized.columns:
            r_value, p_value = pearsonr(df_normalized[feature], y_train_flat)
            print(p_value, r_value)
            if p_value < 0.05 and abs(r_value) > 0.8:
                selected_features.append(feature)
            writer.writerow({'Features': feature, 'P values': p_value, 'R values': r_value})

print("Selected Features:", selected_features)


X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]

kf = KFold(n_splits=3, shuffle=True, random_state=42)
r2_scores = []
auc_scores = []
accuracy_scores = []
num_classes = len(np.unique(y_train))
for train_index, val_index in kf.split(X_train_selected):
    X_train_fold, X_val_fold = X_train_selected.iloc[train_index], X_train_selected.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    rf = RandomForestClassifier()
    rf.fit(X_train_fold, y_train_fold)

    y_val_prob = rf.predict_proba(X_val_fold)

    y_pred = rf.predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, y_pred)
    accuracy_scores.append(accuracy)

    if num_classes > 2:
        auc_value = roc_auc_score(label_binarize(y_val_fold, classes=np.unique(y_train)), y_val_prob, multi_class="ovr")
    else:
        auc_value = roc_auc_score(y_val_fold, y_val_prob[:, 1])

    auc_scores.append(auc_value)

print(auc_scores, accuracy_scores)
