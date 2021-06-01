import numpy as np 
import pandas as pd 
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from lightgbm import LGBMClassifier
import gc

df_train_full = pd.read_csv('train_2_models.csv')
print(df_train_full.columns)
df_test_full = pd.read_csv('test_2_models.csv')
print(df_test_full.columns)

#Drop the unwanted columns
train = df_train_full.drop(['Unnamed: 0', 'image_name'],axis=1)
test = df_test_full.drop(['Unnamed: 0'],axis=1)

"""
#Label Encode categorical features
train.sex.fillna('NaN',inplace=True)
test.sex.fillna('NaN',inplace=True)
train.anatom_site_general_challenge.fillna('NaN',inplace=True)
test.anatom_site_general_challenge.fillna('NaN',inplace=True)
le_sex = LabelEncoder()
le_site = LabelEncoder()
train.sex = le_sex.fit_transform(train.sex)
test.sex = le_sex.transform(test.sex)
train.anatom_site_general_challenge = le_site.fit_transform(train.anatom_site_general_challenge)
test.anatom_site_general_challenge = le_site.transform(test.anatom_site_general_challenge)
"""


# In[18]:


folds = StratifiedKFold(n_splits= 5, shuffle=True)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()
features = [f for f in train.columns if f != 'target']
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train['target'])):
    train_X, train_y = train[features].iloc[train_idx], train['target'].iloc[train_idx]
    valid_X, valid_y = train[features].iloc[valid_idx], train['target'].iloc[valid_idx]
    clf = LGBMClassifier(
        device='gpu',
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=8,
        colsample_bytree=0.5,
        num_leaves=50,
        random_state=23
    )
    print('*****Fold: {}*****'.format(n_fold))
    clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], 
            eval_metric= 'auc', verbose= 20, early_stopping_rounds= 20)

    oof_preds[valid_idx] = clf.predict_proba(valid_X, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del clf, train_X, train_y, valid_X, valid_y
    gc.collect()


# In[19]:

submission = pd.DataFrame({
    "image_name": df_test.image_name, 
    "target": sub_preds
})
submission.to_csv('submission.csv', index=False)

