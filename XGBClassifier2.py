# %% [markdown]
# # Import

# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# %% [markdown]
# # Data

# %%
path = "dataset/train.csv"
df_train = pd.read_csv(path)

path = "dataset/test.csv"
df_test = pd.read_csv(path)

# %%
df_train.info()
df_test.info()

# %%
# print(df_train)
# print(df_test)

# %%
# Check Duplication row in training data
print(df_train.duplicated().sum())

# %%
# All of the rows inside df_test seems to be also inside the df_train
duplicates = pd.merge(df_train, df_test, how='inner')
duplicates

# %%
# # Unique values in each column
# for col in df_train:
#     print(f'+ Unique values in {col}: {df_train[col].unique()}')

# %%
# Number of unique values
df_train.nunique()

# %%
# count each categorical variable frequency
cate_var = df_train.select_dtypes(include=object)
for col in cate_var.columns:
    print(f"{df_train[col].value_counts()}, \n")

# %%
# summary each numerical variable
num_var = df_train.select_dtypes(include=int)
for col in num_var.columns:
    print(f"{df_train[col].describe()}, \n")

# %%
# Merge train and test dataset and Drop Duplicate row 
merge_train_test = pd.concat([df_train, df_test])
new_df = merge_train_test.drop_duplicates()
# new_df

# %%
# Number of yes and no in the response feature, only around 12% yes out of all
counts = new_df['y'].value_counts()
counts

# %% [markdown]
# # Split Data

# %%
X = new_df.iloc[:,:16]
y = new_df.iloc[:,16]

X_train, X_temp, y_train, y_temp = train_test_split(X,y, random_state=42, test_size=0.2, stratify=y)

X_test, X_val, y_test, y_val = train_test_split(X_temp,y_temp, random_state=42, test_size=0.5, stratify=y_temp)

# %% [markdown]
# # Data Preprocessing

# %%
numeric_features = ['age','balance','day','duration','campaign','pdays','previous']

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_features = ['job','marital','education','default','housing',
                        'loan','contact','month','poutcome']

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Fit and transform the training data and transform the val and test data
X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

# %%
# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Percentages of yes in each of train, val, test dataset
print(sum(y_train)/len(y_train))
print(sum(y_val)/len(y_val))
print(sum(y_test)/len(y_test))

label_names = label_encoder.classes_
label_names

# %% [markdown]
# # Models Assessment with default hyperparameters:

# %%
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# %% [markdown]
# ## DecisionTree

# %%
# decisiontreeclf = DecisionTreeClassifier()
# # print(decisiontreeclf.get_params())

# print(cross_val_score(decisiontreeclf, X_train_transformed, y_train, cv=kfold, scoring='f1').mean())

# %% [markdown]
# ## RandomForest

# %%
# randomforestclf = RandomForestClassifier()
# # print(randomforestclf.get_params())

# print(cross_val_score(randomforestclf, X_train_transformed, y_train, cv=kfold, scoring='f1').mean())

# %% [markdown]
# ## AdaBoost

# %%
# adaclf = AdaBoostClassifier(algorithm="SAMME")
# # print(adaclf.get_params())

# print(cross_val_score(adaclf, X_train_transformed, y_train, cv=kfold, scoring='f1').mean())

# %% [markdown]
# ## GradientBoosting

# %%
# gbclf = GradientBoostingClassifier()
# # print(gbclf.get_params())

# print(cross_val_score(gbclf, X_train_transformed, y_train, cv=kfold, scoring='f1').mean())

# %% [markdown]
# ## XGBoost

# %%
xgbclf = xgb.XGBClassifier()
# print(xgbclf.get_params())

print(cross_val_score(xgbclf, X_train_transformed, y_train, cv=kfold, scoring='f1').mean())

# %% [markdown]
# # Preliminary XGBoost Model:

# %%
xgbclf1 = xgb.XGBClassifier(
    objective="binary:logistic",
    seed=42,
    eval_metric="aucpr",
    n_estimators=200,
    early_stopping_rounds=10,
    tree_method='exact',
    booster='gbtree',
)

model1 = xgbclf1.fit(
    X_train_transformed,
    y_train,
    eval_set=[(X_train_transformed, y_train), (X_val_transformed, y_val)],
    verbose=True,
)

# %%
predictions = model1.predict(X_test_transformed)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Will not subscribe', 'Will subscribe'])
disp.plot()
plt.savefig('confusionmatrix1.png', bbox_inches='tight')

# %% [markdown]
# # Hyperparameters tuning

# %%
# # Round One
# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.1, 0.01, 0.05],
#     'gamma': [0, 0.25, 1],
#     'reg_lambda': [0, 1, 10],
#     'scale_pos_weight': [1, 3, 5],
# }

# # Round Twos
# param_grid = {
#     'max_depth': [5,7,9],
#     'learning_rate': [0.1, 0.3, 0.5],
#     'gamma': [0.25],
#     'reg_lambda': [0, 0.25, 0.5],
#     'scale_pos_weight': [3],
# }

# grid_search = GridSearchCV(
#     estimator=xgb.XGBClassifier(
#                                 objective='binary:logistic',
#                                 seed=42,
#                                 # subsample=0.9,
#                                 # colsample_bytree=0.5,
#                                 early_stopping_rounds=10,
#                                 eval_metric='auc',
#                                ),
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=kfold,  
#     verbose=0
# )

# grid_search.fit(X_train_transformed,
#                 y_train,
#                 eval_set=[(X_val_transformed, y_val)],
#                 verbose=False
#                )

# print("Best hyperparameters: ", grid_search.best_params_)

# %%
# with subsample and colsample_bytree
# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'reg_lambda': 0, 'scale_pos_weight': 3}
# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'reg_lambda': 0.25, 'scale_pos_weight': 3}

# without subsample and colsample_bytree
# {'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 5, 'reg_lambda': 0, 'scale_pos_weight': 3}
# {'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 7, 'reg_lambda': 0.25, 'scale_pos_weight': 3}

# %%
xgbclf2 = xgb.XGBClassifier(
    objective='binary:logistic',
    seed=42,
    eval_metric='aucpr',
    n_estimators=200,
    early_stopping_rounds=10,
    tree_method='exact',
    booster='gbtree',
    gamma=0.25,
    learning_rate=0.1,
    max_depth=7,
    reg_lambda=0.25,
    scale_pos_weight=3,
)

model2 = xgbclf2.fit(X_train_transformed,
          y_train,
          eval_set=[(X_train_transformed, y_train), (X_val_transformed, y_val)],
          verbose=True        
)

# %%
predictions = model2.predict(X_test_transformed)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Will not subscribe', 'Will subscribe'])
disp.plot()
plt.savefig('confusionmatrix2.png', bbox_inches='tight')

# %% [markdown]
# ## Feature Importance

# %%
# All Feature names 
feature_names  = preprocessor.get_feature_names_out().tolist()

# Importance score for each feature
importance_score = model2.feature_importances_
# model2.get_booster().get_score(importance_type= 'gain')

# Zip and sort
res = dict(zip(feature_names, importance_score))
sorted_by_values = sorted(res.items(), key=lambda x:x[1])

# Top 10 features
n = 5
top_n_features = sorted_by_values[-n:]
data = dict(top_n_features)
names = list(data.keys())
values = list(data.values())
fig, ax = plt.subplots(figsize=(30, 10))
plt.bar(range(len(data)), values, tick_label=names)
plt.savefig('feature_importance.png', bbox_inches='tight')

# %%
contingency_table = pd.crosstab(new_df.poutcome, new_df.y, margins= False)
contingency_table

# %%
contingency_table.plot(kind='bar', color={'yes': 'blue', 'no': 'orange'}, stacked=True ,rot=0, 
                xlabel='Previous Outcome', ylabel='Number of Outcomes', title='Previous outcome and outcome')
plt.legend(title='Outcome')
plt.savefig('previous_outcome.png', bbox_inches='tight')

# %% [markdown]
# # Final Evaluation

# %%
y_pred = xgbclf1.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# %%
y_pred = xgbclf2.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# %% [markdown]
# # Serialization 

# %%
# xgbclfout = Pipeline(
#     steps=[
#         ('preprocessor', preprocessor),
#         ('model', model2)
#     ]
# )

# joblib.dump(xgbclfout, "my_model.pkl")


