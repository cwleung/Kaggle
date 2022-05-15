import glob
import math
import os
import random
import sys
from itertools import cycle

import catboost as cat
import category_encoders as ce
import lightgbm as lgb
import matplotlib.pylab as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn import (decomposition, ensemble, linear_model, metrics,
                     model_selection, preprocessing, tree)
from tqdm import tqdm


# Handling missing data
def fill_missing(df, default_val=0):
    df.fillna(0, inplace=True)
    return df

# Imputations
def imputate_missing():
    pass


# Features filtering
def drop_columns(df, max_cat=10, max_num=50):
    # categorical features
    cat_cols = df.select_dtypes('object')
    single_cat = np.where(cat_cols.nunique()==1)
    # many cat
    large_cat = np.where(cat_cols.nunique()>max_cat)

    # numerical features
    num_cols = df.select_dtypes(['int', 'float'])
    # single num
    single_num = np.where(num_cols.nunique()==1)
    # many num
    large_num = np.where(num_cols.nunique()>max_num)

    drop_cols = np.concatenate((single_num, large_num,single_cat,large_cat),axis=1)
    drop_cols = drop_cols.squeeze()

    df_proc = [col for col in df.columns if col not in drop_cols]
    return df_proc



# label encoding those cat columns
def label_encoding(train_df, test_df, columns):
    for col in columns:
        le = preprocessing.LabelEncoder()
        # combine value
        values = train_df[col].append(test_df[col])
        le.fit(values)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    return train_df, test_df

def run_lgbm(train_df, train_cols, target_col, params):
    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_predictions = []

    oof_predictions_proba = np.zeros(len(train_df))

    for idx, (train_idx, valid_idx) in enumerate(skf.split(train_df, train_df[target_col])):

        X_train = train_df.iloc[train_idx][train_cols]
        y_train = train_df.iloc[train_idx][target_col]

        X_valid = train_df.iloc[valid_idx][train_cols]
        y_valid = train_df.iloc[valid_idx][target_col]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
                eval_metric='auc',
                verbose=-1
        )

        y_valid_pred = model.predict_proba(X_valid)[:, 1]
        oof_predictions_proba[valid_idx] = y_valid_pred

    score = metrics.roc_auc_score(train_df[target_col], oof_predictions_proba)
    
    return score

    # ROC

    # Hyperparameter tuning

    # Evaluations
    