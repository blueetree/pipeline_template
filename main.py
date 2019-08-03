import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transformers import ColumnExtractor, DFStandardScaler, DFFeatureUnion, DFImputer
from transformers import DummyTransformer, Log1pTransformer, ZeroFillTransformer
from transformers import DateFormatter, DateDiffer, MultiEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

##############################
# Load Data
##############################
Dir_PATH = ''
File_PATH = Dir_PATH + 'filename.csv'
df = pd.read_csv(File_PATH)

##############################
# Quick Look
##############################
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
# Group columns by type of preprocessing needed
OUTCOME = 'permit_status'
NEAR_UNIQUE_FEATS = ['name_of_event', 'year_month_app', 'organization']
NUM_FEATS = ['attendance']
CAT_FEATS = [
    'permit_type', 'event_category', 'event_sub_category',
    'event_location_park', 'event_location_neighborhood']
MULTI_FEATS = ['council_district', 'precinct']  # whats different from cat?
DATE_FEATS = ['application_date', 'event_start_date', 'event_end_date']

##############################
# Data Splitting
##############################
# Set aside 25% as test data
X=df.drop([OUTCOME], axis=1)
Y=df[OUTCOME]
# bins = np.linspace(np.min(OUTCOME), np.max(OUTCOME), 5)
# Y_binned = np.digitize(Y, bins)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30, stratify=Y)
print ("train feature shape: ", X_train.shape)
print ("test feature shape: ", X_test.shape)

##############################
# EDA
##############################
# y_train = y_train.reset_index(drop=True)
# train = pd.concat([X_train, y_train], axis=1)

##############################
# Build Pipeline
##############################
# Pipeline Stacking
pipeline = Pipeline([
    ('features', DFFeatureUnion([
        # ('dates', Pipeline([
        #     ('extract', ColumnExtractor(DATE_FEATS)),
        #     ('to_date', DateFormatter()),
        #     ('diffs', DateDiffer()),
        #     ('mid_fill', DFImputer(strategy='median'))
        # ])),
        ('categoricals', Pipeline([
            ('extract', ColumnExtractor(CAT_FEATS)),
            ('dummy', DummyTransformer())
        ])),
        # ('multi_labels', Pipeline([
        #     ('extract', ColumnExtractor(MULTI_FEATS)),
        #     ('multi_dummy', MultiEncoder(sep=';'))
        # ])),
        ('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer()),
            ('log', Log1pTransformer())
        ]))
    ])),
    ('scale', DFStandardScaler()),
    ('pca', PCA()),
    ('LR', LogisticRegression(random_state=5678))
])

##############################
# Modeling + Tuning
##############################
check_params= {
    'pca__n_components': [2, 4, 6, 8],
    'LR__C': [0.001, 0.1, 1, 10, 20, 50]
}
y_train = np.where(y_train == 'Complete', 1, 0)
y_test = np.where(y_test == 'Complete', 1, 0)
for cv in range(4, 5):
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv, scoring='roc_auc')
    create_grid.fit(X_train, y_train)
    print ("score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test, y_test)))
    print ("!!!!!!!! Best-Fit Parameters From Training Data !!!!!!!!!!!!!!")
    print (create_grid.best_params_)
print ("out of the loop")
print ("grid best params: ", create_grid.best_params_)
# pipeline with dates and multi_labels auc = 0.51
# pipeline without dates and multi_labels auc = 0.73

##############################
# Evaluation
##############################
final_model = create_grid.best_estimator_
final_model.fit(X_train, y_train)

# regression evaluation
# from sklearn.metrics import mean_squared_error
# def my_scorer(estimator, X , y ):
#     prediction = estimator.predict(X)
#     rmse = np.sqrt(mean_squared_error(y, prediction))
#     return rmse
# print ("\n**tuned random forest resgressor**")
# print ("RMSE_train", my_scorer(final_model, X_train , y_train))
# print ("RMSE_test",my_scorer(final_model, X_test , y_test))

# feature_importance = final_model.feature_importances_
# attributes = []
# tup = zip(feature_importance, attributes)
# tup_sorted = sorted(tup, reverse = True)
# for i in range(len(attributes)):
#     print(tup_sorted[i])

# classification evaluation
from sklearn.metrics import roc_auc_score
y_pred = final_model.predict_proba(X_test)[:, 1]
# y_true = np.where(y_test == 'Complete', 1, 0)
auc_test = roc_auc_score(y_test, y_pred)
print(auc_test)
