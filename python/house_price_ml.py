import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import uniform, skew
from scipy.special import boxcox1p, inv_boxcox1p

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit, learning_curve, cross_val_score, KFold
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
import xgboost as xgb
from xgboost import XGBRegressor

from category_encoders.target_encoder import TargetEncoder

from sklearn import metrics

import joblib
import json




train_test_split_random_state = 120
train_test_split_random_state = 230

def display_estimator_parameters_for_pipeline(ppl):
    '''
    Display a formatted list of all the parameters and their values of the estimator of a pipeline.

    Parameters:
    ---
    ppl: The pipeline of interest. The estimator must be the last step in the pipeline.
    '''
    name_model = ppl.steps[-1][0] + "__"
    print("Name: %s" % (ppl.steps[-1][0]))
    print("Estimator: %s" % (ppl.steps[-1][1]))    
    print("------------------------------------------------")
    pars = ppl.get_params()
    modelpars = {x: pars[x] for x in pars if name_model in x}
    for (key, val) in modelpars.items():
        print("%-24s: %-20s" % (key.replace(name_model, ""), str(val)))

def examine_learning_curve(ppl, X, y, n_splits=5):
    '''
    Display the learning curves for a pipeline. The top panel shows the training errors and cross-validation errors as a function of the training sizes (subsample of 10% to 80% of the data). The bottom panel shows the time cost.

    Parameters:
    ---
    ppl: Pipeline. 
    X: Features of the training data
    y: Targets of the training data
    n_splits: Number of train/cross-validation pairs to resample
    '''

    fig, axs = plt.subplots(2, 1, num='learning_curve', figsize=(5,8))
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=12)
    lc_sizes, lc_train, lc_cv, lc_time, _ = \
        learning_curve(ppl, X, y, cv=cv, return_times=True, \
                       train_sizes=np.linspace(0.1, 1.0, 9),
                       scoring = 'neg_root_mean_squared_error')
    # score vs. size
    ax = axs.flatten()[0]
    x, y, dy = lc_sizes, lc_train.mean(axis=1), lc_train.std(axis=1)
    ax.plot(x, y, "bo-", label="training")
    ax.fill_between(x, y-dy, y+dy, alpha=0.2, color='blue')
    x, y, dy = lc_sizes, lc_cv.mean(axis=1), lc_cv.std(axis=1)
    ax.plot(x, y, "ro-", label="cross-validation")
    ax.fill_between(x, y-dy, y+dy, alpha=0.2, color='red')
    ax.legend(loc="lower right")
    ax.set_ylabel("r2 scores")
    ax.set_title("Learning Curve (%s)" % (ppl.steps[-1][0]))
    ax.grid()
    # time vs. size
    ax = axs.flatten()[1]
    x, y, dy = lc_sizes, lc_time.mean(axis=1), lc_time.std(axis=1)
    ax.plot(x, y, "bo-", label="Time")
    ax.set_ylabel("Time cost (sec)")
    ax.set_xlabel("Training Size")
    ax.legend(loc="lower right")
    ax.grid()
    plt.show()

# Feature selection
# Based on result from the explorative data analysis (EDA)
# ================================================================

print("Loading data from train.csv.")
train = pd.read_csv("../data/train.csv")
m = train.shape[0]

# Re-define train_derived
train_derived = train.copy()

# outliers = train[train['GrLivArea'] > 4000].index
# outliers.union(train[train['GarageArea'] > 1200].index)

# outliers = train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index

# ----------------------------------------------------------------
from collections import Counter
num_col = train.loc[:,'MSSubClass':'SaleCondition'].select_dtypes(exclude=['object']).columns
# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 3.0 * IQR ## increased to 1.7
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers 
Outliers_to_drop = detect_outliers(train,2, num_col)
# ----------------------------------------------------------------

train_derived = train_derived.drop(outliers)
#train_derived = train_derived.drop(Outliers_to_drop)

# Dropping outliers improves RMSE by 0.004

# Drop features that will not be used in modeling
# Combines features that would be dropped:
drop_features = ["Id","Utilities","Street"]
drop_features += ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] # sparse features
#drop_features += ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']

for var in drop_features:
    if(var in train_derived):
        # print("Drop feature: ", var)
        train_derived = train_derived.drop(var, axis=1)

#Hard code the ordinal features using a .json file
with open('ordinal.json',) as f:
    ordinal_feature_encoder = json.load(f)
train_derived = train_derived.replace(ordinal_feature_encoder)

# MSSubClass is a nominal (categorical) feature but is encoded with numerics
train_derived['MSSubClass'] = train_derived['MSSubClass'].astype('object')

for col in ['MSSubClass', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']:
    train_derived[col] = train_derived[col].astype('object')

vars_fill_zero = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'MasVnrArea', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars']

ord_features = pd.Index(ordinal_feature_encoder.keys()).join(X_train.columns, how='inner')

for var in vars_fill_zero:
    train_derived[var] = train_derived[var].fillna(0)
for var in ord_features:
    train_derived[var] = train_derived[var].fillna(0)

train_derived['TotalSF'] = train_derived['TotalBsmtSF'] + train_derived['1stFlrSF'] + train_derived['2ndFlrSF']    

# --------------------------------
# train_derived['LotFrontage'] = train_derived.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# https://www.kaggle.com/dmkravtsov/3-2-house-prices

# train_derived['AllSF'] = train_derived['GrLivArea'] + train_derived['TotalBsmtSF']
# train_derived['Area'] = train_derived['LotArea'] * train_derived['LotFrontage']
# train_derived['Area_log'] = np.log1p(train_derived['Area'])
# train_derived['Is_MasVnr'] = [1 if i != 0 else 0 for i in train_derived['MasVnrArea']]
# train_derived['Is_BsmtFinSF1'] = [1 if i != 0 else 0 for i in train_derived['BsmtFinSF1']]
# train_derived['Is_BsmtFinSF2'] = [1 if i != 0 else 0 for i in train_derived['BsmtFinSF2']]
# train_derived['Is_BsmtUnfSF'] = [1 if i != 0 else 0 for i in train_derived['BsmtUnfSF']]
# train_derived['Is_TotalBsmtSF'] = [1 if i != 0 else 0 for i in train_derived['TotalBsmtSF']]
# train_derived['Is_LowQualFinSF'] = [1 if i != 0 else 0 for i in train_derived['LowQualFinSF']]
# train_derived['Is_GarageArea'] = [1 if i != 0 else 0 for i in train_derived['GarageArea']]
# train_derived['Is_WoodDeckSF'] = [1 if i != 0 else 0 for i in train_derived['WoodDeckSF']]
# train_derived['Is_OpenPorchSF'] = [1 if i != 0 else 0 for i in train_derived['OpenPorchSF']]
# train_derived['Is_EnclosedPorch'] = [1 if i != 0 else 0 for i in train_derived['EnclosedPorch']]
# train_derived['Is_3SsnPorch'] = [1 if i != 0 else 0 for i in train_derived['3SsnPorch']]
# train_derived['Is_ScreenPorch'] = [1 if i != 0 else 0 for i in train_derived['ScreenPorch']]


# --------------------------------

# Prepare data for training
y = train_derived['SalePrice']
X = train_derived.drop('SalePrice', axis=1)

num_features = X.select_dtypes(['int', 'float']).columns
#Log transform all skewed features

vars_skewed = []

def transform_skewed_variables(arr, method='log1p'):
    if(method == "log1p"): return np.log1p(arr)
    if(method == "boxcox"): return boxcox1p(arr, 0.15)

method_skewed = 'boxcox'
for col in num_features:
    if(abs(skew(train_derived[col])) > 1.0):
        vars_skewed.append(col)
        # print("%-24s %f" % (col, skew(train_derived[col])))
        train_derived[col] = transform_skewed_variables(train_derived[col], method=method_skewed)
train_derived['SalePrice'] = np.log1p(train_derived['SalePrice'])

# Now separate the training data into training and test set.
# The "test set" here is actually a hold-out cross validation set used to test if our ML models perform well. Later, we split the "train set" into k-Fold train/cross validation sets in order to find the hyperparameters that optimize average scores on the train/cv split. Finally we will use the optimized model on the hold-out set, before we apply it to the "real" test set provided in the test.csv file.
print("Separating train and test (hold-out cross-validation) set.")

# Test with a small set of features
# train_derived = train_derived[['Neighborhood', 'YearBuilt', 'SalePrice']]

X_train, X_test, y_train, y_test = \
    train_test_split(train_derived, train_derived['SalePrice'], test_size=0.3, random_state=train_test_split_random_state)

# X_train, X_test, y_train, y_test = \
#     train_test_split(train, train['SalePrice'], test_size=0.3, random_state=120)

X_train = X_train.drop('SalePrice', axis=1)
X_test = X_test.drop('SalePrice', axis=1)

# te = TargetEncoder(cols = ['Neighborhood'])
# te.fit(X_train, y_train)
# X_train = te.transform(X_train)
# X_test = te.transform(X_test)

# Caveat: 
# An error raises if one passes DataFrame to the ColumnTransformer:
# "No valid specification of the columns. Only a scalar, list or slice of all integers or all strings, or boolean mask is allowed"
# Instead, one should pass indices (DataFrame.columns)

num_features = X_train.select_dtypes(['int', 'float']).columns
cat_features = X_train.select_dtypes('object').columns

from sklearn.svm import SVR
from lightgbm import LGBMRegressor

model_lr = LinearRegression() # Toy model
model_sgd = SGDRegressor() # Linear Regressor with Stochastic Gradient Descent
model_ridge = Ridge(alpha=10.0)
model_lasso = Lasso(alpha=0.0003, max_iter=1.e5)
model_rf = RandomForestRegressor(n_estimators=400, max_features='sqrt', max_depth=100, min_samples_split=2, min_samples_leaf=1) # Random Forest Decision Tree
model_svr = SVR(C=20, epsilon=0.008, gamma=0.0003)
model_lgbm = LGBMRegressor(objective='regression', num_leaves=6, learning_rate=0.01, n_estimators=1000, max_bin=200, bagging_fraction=0.8, bagging_freq=4, bagging_seed=8, feature_fraction=0.2, feature_fraction_seed=8, min_sum_hessian_in_leaf=11, verbose=-1, random_state=42)
model_xgb = XGBRegressor(n_estimators=800, min_child_weight=3, max_depth=3, gamma=0, learning_rate=0.08, subsample=0.7, reg_lambda=0.1, reg_alpha=0.0, colsample_bytree=1.0) # Extreme Gradient Boosting Decision Tree

# Best-fit values from https://www.kaggle.com/goldens/house-prices-on-the-top-with-simpl
# model_xgb = XGBRegressor(colsample_bylevel = 0.6971640307744038, colsample_bytree = 0.6222251306452113, gamma = 0.045557016198868025, learning_rate = 0.012334455088591312, max_depth = 4, n_estimators = 2100, reg_lambda = 2.1713829044702413, subsample = 0.8459257473555255)

# VotingRegressor()

# Note: Sklearn UG says SGD is more suitable for large data (m > 10,000), otherwise use Lasso or Ridge

print("Preparing the data for both numeric and categorical data.")
# Numeric features
# ----------------
# Imputer: SimpleImputer(missing_values=np.nan, strategy={"mean","median","constant"})
# Fill missing values with mean() or median() of the given columns, or constant value specified by fill_value
#
# Scaler: StandardScaler() standardizes continuous variable to (mean, std) = (0.,1.)

ppl_num = Pipeline(steps = \
    [('imp_num', SimpleImputer(strategy = 'median')), \
     ('scaler', RobustScaler())] \     
     # ('scaler', StandardScaler())] \
)

# Categorical features
# ----------------
# Imputer: SimpleImputer(missing_values=np.nan, strategy={"constant", "most_frequent"})
# Fill missing values with the most frequent category, or a constant specified by fill_value 
# 
# Encoder: OneHotEncoder() (OHE) "expands" categorical features into an array of booleans
# For example:
#   - Input: 2 features {gender in ["male", "female"], city in ["Boston", "London", "Paris"]}
#   - Output: 5 features {male in [0,1], female in [0,1], Boston in [0,1], ...}
#   - {gender:male, city:London} becomes {1, 0, 0, 1, 0}
# Useful arguments: handle_unknown = 'ignore'

ppl_cat = Pipeline(steps = \
    [
    # ('imp_cat', SimpleImputer(strategy='constant', fill_value='none')), \
    ('imp_cat', SimpleImputer(strategy='most_frequent')), \
    ('ohe', OneHotEncoder(handle_unknown='ignore'))] \
)

ct = ColumnTransformer(transformers= \
    [('numeric', ppl_num, num_features), \
     ('categorical', ppl_cat, cat_features)] \
)

models = [model_lr, model_ridge, model_lasso, model_sgd, model_rf, model_xgb]

# In a jupyter environment, setting display to "diagram" will show the execution plan of a pipeline visually
set_config(display="diagram")

# Now train models with the different ML algorithms using default parameters
def train_models(models, show_params=False):
    for model in models:
        ppl_full = Pipeline(steps = [('ct', ct), ('model', model)])
        ppl_full.fit(X_train, y_train)
        pred = ppl_full.predict(X_test)

        if(show_params):
            display_estimator_parameters_for_pipeline(ppl_full)
        score_mse = metrics.mean_squared_error(pred, y_test)
        score = ppl_full.score(X_test, y_test)

        print("RMSE (%s) = %f" % (str(model), np.sqrt(score_mse)))
        print("Score (%s) = %f" % (str(model), score))
        print("----------------")
    return ppl_full

 # Explore the Hyperparameter Space

# ================================================================

# GridSearchCV parameters:
# ----------------
# n_jobs = -1: Use all processors available
# cv = 3: Use 3-fold cross-validation splitting. Calls (Stratified)KFold

# Now some very useful method to check the GridSearchCV results:
# cv_results_: Test results matrix
# best_params_: Best set of parameters

def train_model_lasso_parameters_search(auto=True):
    ppl_lasso = Pipeline(steps = [('ct', ct), ('lasso', model_lasso)])

    if(auto): # Ues GridSearchCV to automatically search for optimal parameters
        grid_params = {'lasso__alpha':[1.e-5, 3.e-5, 0.0001, 0.0003, 0.001, 0.003, 0.1, 0.3]}
        reg_lasso = GridSearchCV(ppl_lasso, grid_params, \
                                 cv=5, verbose=3, n_jobs=-1, \
                                 scoring = 'neg_root_mean_squared_error',
                                 return_train_score=True) # r2
        model = reg_lasso.fit(X_train, y_train)
        gcv_matrix = pd.DataFrame(model.cv_results_)
        gcv_bestpar = model.best_params_
        print(gcv_matrix.groupby('param_lasso__alpha')[['mean_fit_time', 'mean_train_score', 'mean_test_score']].mean())
        print()        
        print("Best Parameters:")
        print("--------------------------------")
        for (key, val) in gcv_bestpar.items():
            print("%-24s: %-20s" % (key, str(val)))
        print()
        print("Best Fit Coefficients:")
        print("--------------------------------")
        reg = reg_lasso.best_estimator_.named_steps['lasso']
        print("Mean: %f" % (abs(reg.coef_).mean()))
        print("Std : %f" % (abs(reg.coef_).std()))
        print()
        print("Best Score (Hold-out): %7.5f" % (reg_lasso.score(X_test, y_test)))
        # Note: I found that the score from using the hold-out cross validation set could be much lower than the mean_test_score (or best_score_). The reason is likely due to high variance of the model. Using different values of random_state in the train_test_split() can lead to large fluctuations of the best scores from the GridSearchCV as well as the test score on the hold-out CV data. In addition, since the best scores from the GridSearchCV averages over several folds of a larger sample (X_train, 70% of the full sample), they are likely to be more constant and higher than the hold-out scores.
        return model, reg_lasso.best_estimator_
    else:
        # Manually choose parameters and examine their performances
        # Use the same train/cv split
        alphas = [0.3, 1.0, 3.0, 10., 30., 100., 300., 1000.]
        scores_train, scores_cv = [], []
        for alp in alphas:
            print("-------- alpha = %f --------" % (alp))
            ppl_lasso.set_params(lasso__alpha=alp, lasso__max_iter = 10000)
            # Default lasso__max_iter = 1000 but it complains that "Objective did not converge". After using max_iter = 10000, results converge at alpha >= 10.0
            ppl_lasso.fit(X_train, y_train)
            scores_train.append(ppl_lasso.score(X_train, y_train))
            scores_cv.append(ppl_lasso.score(X_test, y_test))
        plt.plot(alphas, scores_train, "b.-")
        plt.plot(alphas, scores_cv, "r.-")
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_xlabel("alpha")        
        ax.set_ylabel("r2 score")
        ax.set_xticks(alphas)
        ax.set_xticklabels([alp for alp in alphas])
        return _, _

# Model: Random Forest Regressor
def train_model_rf_parameters_search(auto=True, par=None, ax=None):
    ppl_rf = Pipeline(steps = [('ct', ct), ('rf', model_rf)])

    if(auto):
        grid_params = {'rf__n_estimators':[100, 400], \
                       'rf__max_features':['auto', 'sqrt'], \
                       'rf__max_depth':[10, None], \
                       'rf__min_samples_split':[5, 20], \
                       'rf__min_samples_leaf':[1, 2]}
        # Critical Parameters of a random forest algorithm:
        # ----------------------------------------------------------------
        # max_features: How to determine the maximum number of features that a random tree resamples? 'auto': All features (Note: For RandomTreeClassifier, 'auto' uses 'sqrt'. Using all features is equivalent to bagging. The disadvantage here is that bagging averages over correlated trees, because certain features might dominates earlier steps of splitting.). 'sqrt' (default), 'log2': in terms of total number of features
        # n_estimators: Number of random trees. Averaging over many random trees helps reducing noice (variance). Using a large n_estimator usually does not hurt but performance will saturate at some value.
        # max_depth: If 'None', split the tree until all leaves are pure
        # min_samples_split: Each node should have less samples than this value
        # min_samples_leaf: Each leaf should have more samples than this value. Along with min_samples_split, they adjust the bias/variance trade-off.

        reg_rf = GridSearchCV(ppl_rf, grid_params, \
                              cv=3, verbose=3, n_jobs=-1, \
                              scoring = 'r2', return_train_score=True) # r2
        model = reg_rf.fit(X_train, y_train)
        gcv_matrix = pd.DataFrame(model.cv_results_)
        gcv_bestpar = model.best_params_
        gcv_matrix.fillna(-1) # rf__max_depth could be None
        for par in grid_params:
            print(gcv_matrix.groupby("param_"+par)[['mean_fit_time', 'mean_train_score', 'mean_test_score']].mean())
        print()        
        print("Best Parameters:")
        print("--------------------------------")
        for (key, val) in gcv_bestpar.items():
            print("%-24s: %-20s" % (key, str(val)))
        print()
        print("Best Score (Hold-out): %7.5f" % (reg_rf.score(X_test, y_test)))
        return model, reg_rf
    else: # manually examine parameters one by one
        # Best-fit parameters
        params = {'rf__n_estimators':[50, 100, 200, 400, 800, 1600, 3200], \
                  'rf__max_depth':[5, 10, 20, 50, 100, 200, 300, 400], \
                  'rf__min_samples_split':[2, 5, 10, 15, 20], \
                  'rf__min_samples_leaf':[1, 2, 5, 10, 15, 20]}
        cpad = sns.color_palette() # The color pad
        scores, stds = [], []
        parlst = params["rf__"+par]
        print("Training: ")
        best_score = -1.0
        best_model = None
        for parval in parlst:
            ppl_rf.set_params(**{"rf__"+par:parval})
            cvs = cross_val_score(ppl_rf, X_train, y_train, cv=5, scoring='r2')
            print("%s = %s: %7.5f" % (par, str(parval), cvs.mean()))
            scores.append(cvs.mean())
            stds.append(cvs.std())
            if(cvs.mean() > best_score):
                best_score = cvs.mean()
                best_model = parval
        print()
        print("Best Model: %s = %s" % (par, str(best_model)))
        print("Best Score: %7.5f" % (best_score))
        if(ax == None):
            fig, ax = plt.subplots(1, 1, figsize=(6,4))
        ax.plot(parlst, scores, ".-", color="blue")
        scores = np.array(scores)
        stds = np.array(stds)
        ax.fill_between(parlst, scores-stds, scores+stds, color="blue", alpha=0.2)
        ax.set_xlabel(par)
        ax.set_ylabel("r2 score")
        ax.set_xticks(params["rf__"+par])
        ax.set_xticklabels([parval for parval in parlst])
        return ppl_rf, ppl_rf.named_steps["rf"]
        
        # scores_train, scores_cv = [], []
        # parlst = params["rf__"+par]
        # for parval in parlst:
        #     print("-------- %s = %s --------" % (par, str(parval)))
        #     ppl_rf.set_params(**{"rf__"+par:parval})
        #     ppl_rf.fit(X_train, y_train)
        #     scores_train.append(ppl_rf.score(X_train, y_train))
        #     scores_cv.append(ppl_rf.score(X_test, y_test))
        # plt.plot(parlst, scores_train, "b.-")
        # plt.plot(parlst, scores_cv, "r.-")
        # ax = plt.gca()
        # ax.set_xlabel(par)
        # ax.set_ylabel("r2 score")
        # ax.set_xticks(params["rf__"+par])
        # ax.set_xticklabels([parval for parval in parlst])
        # return _, _

def train_model_ridge_parameters_search():
    ppl_ridge = Pipeline(steps = [('ct', ct), ('ridge', model_ridge)])
    
    grid_params = {'ridge__alpha':[0.3, 1.0, 3.0, 10., 30., 100.]}
    reg_ridge = GridSearchCV(ppl_ridge, grid_params, cv=5, verbose=3, n_jobs=-1, \
                             scoring = 'r2') # r2
    model = reg_ridge.fit(X_train, y_train)
    gcv_matrix = pd.DataFrame(model.cv_results_)
    gcv_bestpar = model.best_params_
    print(gcv_matrix.groupby('param_ridge__alpha')[['mean_fit_time', 'mean_test_score']].mean())
    print("Best Parameters:")
    print("--------------------------------")
    for (key, val) in gcv_bestpar.items():
        print("%-24s: %-20s" % (key, str(val)))
    print("Best Fit Coefficients:")
    print("--------------------------------")
    reg = reg_ridge.best_estimator_.named_steps['ridge']
    print("Mean: %f" % (abs(reg.coef_).mean()))
    print("Std : %f" % (abs(reg.coef_).std()))
    print()
    print("Best Score (Hold-out): %7.5f" % (reg_ridge.score(X_test, y_test)))
    return model, reg_ridge

def train_model_sgd_parameters_search(method='grid'):
    ppl_sgd = Pipeline(steps = [('ct', ct), ('sgd', model_sgd)])
    
    # Check available parameters with SGDRegressor().get_params()
    # alpha = 0.0001: Regularization factor
    # learning_rate = 'invscaling': Other choices are 'constant', 'optimal', 'adaptive'
    # eta0 = 0.01: Initial learning rate

    if(method == 'grid'):
        grid_params = {'sgd__eta0':[0.001, 0.005, 0.01, 0.05, 0.1], \
                       'sgd__alpha':[0.00005, 0.0001, 0.0002], \
                       'sgd__penalty':['l2', 'l1'], \
                       'sgd__learning_rate':['invscaling', 'adaptive'] \
        }
        # note: uniform() generates uniform distribution on [loc, loc + scale], call uniform.pdf(x)
        # note: When the estimator is used in a pipeline, use estimator__paramter
        reg_sgd = GridSearchCV(ppl_sgd, grid_params, cv=3, verbose=3, n_jobs=-1, \
                               scoring = 'r2')
        # Other scoring metrics to consider:
        #   - explained_variance
        #   - neg_root_mean_squared_error
    if(method == 'random'):
        rand_params = {'sgd__alpha':uniform(loc=0.00001, scale=0.00020), \
                       'sgd__penalty':['l2', 'l1'], \
                       'sgd__learning_rate':['invscaling', 'adaptive'] \
        }
        # note: uniform() generates uniform distribution on [loc, loc + scale], call uniform.pdf(x)
        # note: When the estimator is used in a pipeline, use estimator__paramter
        reg_sgd = RandomizedSearchCV(ppl_sgd, rand_params, n_iter=50, random_state=10)
        # n_iter = 10: Number of iterations

    model = reg_sgd.fit(X_train, y_train)
    return model, reg_sgd

def check_sgd_grid_search_matrix(model, plot=False):
    gcv_matrix = pd.DataFrame(model.cv_results_)
    gcv_bestpar = model.best_params_
    
    print("Result from parameters grid search: ")
    print("----------------------------------------------------------------")
    print(gcv_matrix.groupby('param_sgd__eta0')[['mean_fit_time', 'mean_test_score']].mean())
    print(gcv_matrix.groupby('param_sgd__alpha')[['mean_fit_time', 'mean_test_score']].mean())
    print(gcv_matrix.groupby('param_sgd__penalty')[['mean_fit_time', 'mean_test_score']].mean())
    print()
    print("Best Parameters:")
    print("--------------------------------")
    for (key, val) in gcv_bestpar.items():
        print("%24s: %20s" % (key, str(val)))
    print()
    print("Best Score: %7.5f" % (reg_sgd.score(X_test, y_test)))
    
    if(plot):
        g = sns.FacetGrid(gcv_matrix, row='param_sgd__learning_rate', aspect=2.0)
        g.map_dataframe(sns.pointplot, x='param_sgd__eta0', y='mean_fit_time', \
                        hue='param_sgd__penalty' \
        )
        g.set(ylim=(0.0, 2.0))
        g.set_ylabels("Fit Time")
        g.set_xlabels("Initial Learning Rate")
        g.fig.subplots_adjust(left=0.15)
        g.axes.flatten()[0].legend(loc='lower left')

        g = sns.FacetGrid(gcv_matrix, row='param_sgd__learning_rate', aspect=2.0)
        g.map_dataframe(sns.pointplot, x='param_sgd__eta0', y='mean_test_score', \
                        hue='param_sgd__penalty' \
        )
        g.set_ylabels("Test Score")
        g.set_xlabels("Initial Learning Rate")
        g.fig.subplots_adjust(left=0.15)
        g.axes.flatten()[0].legend(loc='lower left')
        plt.show()

# XGBoost:
# ================================================================
# A few Notes on this algorithm. XGBoost has become one of the most popular ML alorithms on kaggle due to its performance and flexibility. It is similar to the gradient boosted decision tree (GBDT or GBT) but has several novel features that in many circumstances prove to greatly enhance the performance.

# The following text may be a bit technical. It is kind of my study notes on the referenced paper by D. Nielsen.

# What is a gradient boosted tree algorithm? First of all, observe that a decision tree is essentially as a function T(x) that predicts the outcome y=T(x) given x. Remember that the random forest is an algorithm that averages over many decision trees {T}(x) to reduce variance, with each tree T_i(x) built independently, using different sub-samples of features and/or data. Instead, the GBDT builds a sequence of (small) trees, with each new tree improving upon the results from previous trees. The final tree GBDT(x) predicts the outcome by adding the predictions from all previous trees together, i.e., GBDT(x)=T_0(x)+T_1(x)+.... The concept of using an additive model like the GBDT is called boosting. It is important to note that, while in a random forest all trees are optimized towards predicting Y, in a boosted "forest", each tree has a different optimization goal. Only when they are added together the function predicts Y.

# Boosting can be interpreted as an iterative gradient descent algorithm in the function space. This is a somewhat uncommon way of thinking about gradient descent. Normally, one often applies gradient descent in the parameter space, e.g., during a linear regression. In a parameter space, at each step one updates the parameters by adding a small amount to them in the direction of the negative gradient, where the lost function drops most sharply. Similarly, in a function space, at each step one updates the predictive model (basis function) by adding a "small" function to the existing model in a way that maximizes the gains in the lost function. Therefore, at each iteration, the new model "learns" from the previous model and becomes better. The final model F_k(x), after k iterations, is therefore equals to the sum of all the basis functions trained over the process, f_0(x)+f_1(x)+...+f_k(x).

# How does XGBoost or GBDT combine the boosting algorithm with the tree algorithm? Use the trees T_i(x) as the basis functions f_i(x)! Now one faces two problems: 1. How to find the optimization goal for every f_i(x) at each iteration. 2. How to construct f_i(x), i.e., to build a tree structure that maximize the drop in the lost function.

# For the first task, one need to find which f_i(x) will lead to fastest descent of the lost function L(y, F_{i-1}(x)+f_i(x)). One can approximate the lost function with 1st order ("gradient boosting method") or 2nd order ("Newton boosting method", used by XGBoost) Taylor expansion. The optimization goal of f_i(x) can be found by setting the derivative of the lost function dL/df to zero. Note that, if the lost function is defined by MSE (mean squared error), the gradient (1st order term) is simply the residual from the last best model (y_i - F_{i-1}(x))! The second task is simply applying tree-splitting algorithm given certain constraints such as the maximum size of the tree, etc.

# Finally, what makes XGBoost different from the popular gradient boosted trees?
# 1. The XGBoost uses 2nd-order approximation to the lost function, while the GBDT uses 1st-order approximation. Though higher-order approximations are more accurate, they are less flexible with the choices of the lost function.
# 2. The XGBoost adds a novel regularization term based on the weights of terminal nodes. This help controls bias/variance offset and enables feature selection. In most applications, the weights are the constants that the nodes predict.
# 3. The XGBoost implementation improves upon parallelism.
# 4. The XGBoost handles NULL values better.

# When in doubt, use XGBoost. --- Owen Zhang

# Ref: Didrik Nielsen, Tree Boosting With XGBoost - Why Does XGBoost Win "Every" Machine Learning Competition?

# xgboost main parameters:
# n_estimators: Unlike random forests, a value too large may lead to over-fitting
# learning_rate (eta): It shrinks the feature weights by a constant factor.
# max_depth: The maximum depth of each tree
# min_child_weight: Lower limit of the summed weights on a leaf node. Prevent further tree-spliting if below this limit. Controls bias/variance.
# mean_split_loss (gamma): Minimum loss reduction required for spliting a node
# lambda: L2 regularization coefficient, default = 1
# alpha: L1 regularization coefficient, default = 0.
# subsample: Fraction of data to subsample before each boosting step



# The following article provides an example of tuning XGB parameters.
# The conclusion, though, is that "A significant jump can be obtained by other methods like feature engineering, creating ensemble of models, stacking, etc", instead of by "just using parameter tuning or slightly better models"
# Ref: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


def train_model_xgb_parameters_search(auto=True, par=None, ax=None):
    ppl_xgb = Pipeline(steps = [('ct', ct), ('xgb', model_xgb)])
    if(auto): # automatic grid search for optimal parameters
        # Note: How to use xgboost.cv with sklearn.Pipeline?
        grid_params = { \
                        'xgb__min_child_weight': [1, 5, 10], \
                        'xgb__n_estimators': [200, 400, 600, 800], \
                        'xgb__max_depth': [3,5,7,9], \
                        'xgb__subsample': [0.5, 0.7, 0.9], \
                        'xgb__gamma': [0.1, 0.5, 1, 1.5, 2], \
                        'xgb__learning_rate': [0.05, 0.1] \
        }
        reg_xgb = GridSearchCV(ppl_xgb, grid_params, \
                               cv = 3, n_jobs = -1, verbose = 2, \
                               scoring = 'neg_root_mean_squared_error')
        model = reg_xgb.fit(X_train, y_train)
        return model, reg_xgb
    else:
        # Manually choose parameters and examine their performances
        # Default parameters
        ppl_xgb.set_params(xgb__n_estimators = 400, xgb__min_child_weight=5, xgb__max_depth=2, xgb__gamma = 0, xgb__learning_rate = 0.08, xgb__subsample = 0.7, xgb__reg_lambda=1.0, xgb__reg_alpha=0.3)
        # Use the same train/cv split
        params = { \
                   # 'xgb__min_child_weight': [1, 2, 3, 5, 8, 10], \
                   'xgb__min_child_weight': [1, 10, 100, 1000], \
                   'xgb__n_estimators': [20, 50, 100, 200, 400, 800, 1600, 3200], \
                   'xgb__max_depth': [2, 3, 5, 7, 9], \
                   'xgb__subsample': [0.3, 0.5, 0.7, 0.9], \
                   'xgb__gamma': [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, ], \
                   'xgb__learning_rate': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32], \
                   'xgb__reg_lambda': [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.], \
                   'xgb__reg_alpha': [0.0, 0.1, 0.3, 1.0, 3.0]
        }
        cpad = sns.color_palette() # The color pad
        scores, stds = [], []
        parlst = params["xgb__"+par]
        print("Training: ")
        best_score = -1.0
        best_model = None
        for parval in parlst:
            ppl_xgb.set_params(**{"xgb__"+par:parval})
            # cvs = cross_val_score(ppl_xgb, X_train, y_train, cv=5, scoring='r2')
            cvs = cross_val_score(ppl_xgb, X, y, cv=5, scoring='r2')
            print("%s = %s: %7.5f" % (par, str(parval), cvs.mean()))
            scores.append(cvs.mean())
            stds.append(cvs.std())
            if(cvs.mean() > best_score):
                best_score = cvs.mean()
                best_model = parval
        print()
        print("Best Model: %s = %s" % (par, str(best_model)))
        print("Best Score: %7.5f" % (best_score))
        if(ax == None):
            fig, ax = plt.subplots(1, 1, figsize=(6,4))
        ax.plot(parlst, scores, ".-", color="blue")
        scores = np.array(scores)
        stds = np.array(stds)
        ax.fill_between(parlst, scores-stds, scores+stds, color="blue", alpha=0.2)
        ax.set_xlabel(par)
        ax.set_ylabel("r2 score")
        ax.set_xticks(params["xgb__"+par])
        ax.set_xticklabels([parval for parval in parlst])
        return ppl_xgb, ppl_xgb.named_steps["xgb"]

# ppl_full = train_models([model_lr, model_sgd])
# ppl_full = train_models(models)

# Score for Linear Regression is: 0.253 (bad)
# Score for vanilla SGD regressor is: 0.675 (okay let's start from this)

# model, reg_sgd = train_model_sgd_parameters_search(method='grid')
# model, ppl_lasso = train_model_lasso_parameters_search(auto=True)
# model, ppl_ridge = train_model_ridge_parameters_search()

# rf_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
# fig, axs = plt.subplots(2,2,figsize=(8,6))
# axs = axs.flatten()
# for i, par in enumerate(rf_params):
#     ppl, reg = train_model_rf_parameters_search(auto=False, par=par, ax=axs[i])

# xgb_params = ['n_estimators', 'min_child_weight', 'max_depth', 'gamma', 'subsample', 'learning_rate', 'reg_lambda', 'reg_alpha']
# fig, axs = plt.subplots(2,4,figsize=(10,8))
# axs = axs.flatten()
# for i, par in enumerate(xgb_params):
#     ppl, reg = train_model_xgb_parameters_search(auto=False, par=par, ax=axs[i])
# plt.show()

# CV accuracy is more sensitive to n_estimators, max_depth, learning_rate, subsample, reg_lambda within the parameter range.

def fit_model(model, X_train, y_train, X_test=None, y_test=None, modelname=None):
    ppl = Pipeline(steps = [('ct', ct), ('model', model)])
    ppl.fit(X_train, y_train)
    if(X_test is not None):
        y_pred = ppl.predict(X_test)
        if(modelname): print(modelname)
        print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print(metrics.r2_score(y_test, y_pred))    
    return ppl

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

model_stack = StackingRegressor(estimators=[('ridge', model_ridge), \
                                            ('lasso', model_lasso),\
                                            ('svr', model_svr),\
                                            ('lgbm', model_lgbm),\
                                            ('rf', model_rf),\
                                            ('xgb', model_xgb)],\
                                cv = kfolds)

X = train_derived.drop('SalePrice', axis=1)
y = train_derived['SalePrice']

# ppl_ridge = fit_model(model_ridge, X, y, "Ridge")
# ppl_lasso = fit_model(model_lasso, X, y, "Lasso")
# ppl_xgb = fit_model(model_xgb, X, y, "XGBoost")
# ppl_svr = fit_model(model_svr, X, y, "SVR")
# ppl_rf = fit_model(model_rf, X, y, "Random Forest")
# ppl_lgbm = fit_model(model_lgbm, X, y, "Light GBM")
# ppl_stack = fit_model(model_stack, X, y, "Stack")

def cv_rmse(model, X=X):
    ppl = Pipeline(steps = [('ct', ct), ('model', model)])
    rmse = np.sqrt(-cross_val_score(ppl, X, y,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds))
    return (rmse)

# rmse = cv_rmse(ppl_xgb, X=X).mean()
# cv_rmse(model_ridge, Xtrain, ytrain, "Ridge")
# cv_rmse(model_svr, Xtrain, ytrain, "SVR")
# cv_rmse(model_rf, Xtrain, ytrain, "Random Forest")
# cv_rmse(model_xgb, Xtrain, ytrain, "XGBoost")
# cv_rmse(model_lgbm, Xtrain, ytrain, "Light GBM")
# cv_rmse(model_lasso, Xtrain, ytrain, "Lasso")

# xgb.plot_importance(ppl)
# ppl.feature_importance_

# Re-fit with the best parameters and examine the learning curves, e.g., how does the performance scales with the size of training data?

#examine_learning_curve(ppl_xgb, X_train, y_train)


# Some results from my test:
# - fit_time scales with n_estimators
# - fit_time is an order of magnitude higher with max_features = 'auto'
# - Larger min_samples_split and min_samples_leaf increases bias but reduces fit_time
# - Performance increases rapidly with max_depth but flattens out after max_depth > 10

#check_sgd_grid_search_matrix(reg_sgd)

# model = train_model_xgb_grid_search()

# Save model for later use:
# joblib.dump(model, "model_xgb_gsc.dat")

# perm_result_train = permutation_importance(ppl_lasso, X_train, y_train, n_repeats=10, random_state=24)
# perm_result_test = permutation_importance(ppl_lasso, X_test, y_test, n_repeats=10, random_state=24)
# perm_sorted_idx = perm_result.importances_mean.argsort()[::-1]

def show_important_features(perm_result_train, perm_result_test, X):
    '''
    Display the most important features calculated from the permutation method.

    Parameters:
    ---
    perm_result_train: The output from sklearn.inspection.permutation_importance on the training set
    perm_result_test: On the test set
    X: The training/test data before applying any pipeline. It is used to extract the features.
    '''
    types = ['N/A'] * X.columns.size
    feature_importance = pd.DataFrame({'mean_train':perm_result_train.importances_mean, \
                                       'std_train':perm_result_train.importances_std, \
                                       'mean_test':perm_result_test.importances_mean, \
                                       'std_test':perm_result_test.importances_std, \
                                       'type':types}, \
                                      index=X.columns)
    feature_importance.loc[num_features, 'type'] = 'numerical'
    feature_importance.loc[cat_features, 'type'] = 'nominal'
    feature_importance.loc[ord_features, 'type'] = 'ordinal'
    feature_importance.sort_values(by='mean_train', ascending=False, inplace=True)

    print("The top 10 Numerical features:")
    print(feature_importance[feature_importance['type']=='numerical'][:10])
    print()
    print("The top 5 Nominal features (nominal):")
    print(feature_importance[feature_importance['type']=='nominal'][:5])
    print()
    print("The top 5 Nominal features (ordinal):")
    print(feature_importance[feature_importance['type']=='ordinal'][:5])

    sns.set(font_scale = 1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), num='feature_importance')
    sns.barplot(x = feature_importance['mean_train'][:20], \
                y = feature_importance.index[:20], \
                hue=feature_importance['type'][:20], \
                ax=ax1, orient='h', dodge=False)
    
    feature_importance.sort_values(by='mean_test', ascending=False, inplace=True)
    
    sns.barplot(x = feature_importance['mean_test'][:20], \
                y = feature_importance.index[:20], \
                hue=feature_importance['type'][:20], \
                ax=ax2, orient='h', dodge=False)
    ax1.set_xscale("log")
    ax2.set_xscale("log")

    fig.subplots_adjust(left=0.2, wspace=0.30)
    plt.show()

# show_important_features(perm_result_train, perm_result_test, X)


ct.fit(X)
Xtrain = ct.transform(X)
model_xgb.fit(Xtrain, y)
model_lasso.fit(Xtrain, y)
model_ridge.fit(Xtrain, y)
model_rf.fit(Xtrain, y)
model_svr.fit(Xtrain, y)
#model_lgbm.fit(Xtrain, y)
model_stack.fit(Xtrain, y)

def predict():
    test = pd.read_csv("../data/test.csv")
    testId = test['Id']
    Xtest = test.copy()
    Xtest = Xtest.replace(ordinal_feature_encoder)
    # Xtest['MSSubClass'] = Xtest['MSSubClass'].astype('object')
    for col in ['MSSubClass', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']:
        train_derived[col] = train_derived[col].astype('object')
    num_features = Xtest.select_dtypes(['int', 'float']).columns
    for var in drop_features:
        if(var in Xtest):
            Xtest = Xtest.drop(var, axis=1)
    for col in vars_skewed:
        if(col in Xtest.columns):
            Xtest[col] = transform_skewed_variables(Xtest[col], method=method_skewed)
    for var in vars_fill_zero:
        Xtest[var] = Xtest[var].fillna(0)
    for var in ord_features:
        train_derived[var] = train_derived[var].fillna(0)
        
    Xtest['TotalSF'] = Xtest['TotalBsmtSF'] + Xtest['1stFlrSF'] + Xtest['2ndFlrSF']

    # Xtest['LotFrontage'] = Xtest.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # Xtest['AllSF'] = Xtest['GrLivArea'] + Xtest['TotalBsmtSF']
    # Xtest['Area'] = Xtest['LotArea'] * Xtest['LotFrontage']
    # Xtest['Area_log'] = np.log1p(Xtest['Area'])
    # Xtest['Is_MasVnr'] = [1 if i != 0 else 0 for i in Xtest['MasVnrArea']]
    # Xtest['Is_BsmtFinSF1'] = [1 if i != 0 else 0 for i in Xtest['BsmtFinSF1']]
    # Xtest['Is_BsmtFinSF2'] = [1 if i != 0 else 0 for i in Xtest['BsmtFinSF2']]
    # Xtest['Is_BsmtUnfSF'] = [1 if i != 0 else 0 for i in Xtest['BsmtUnfSF']]
    # Xtest['Is_TotalBsmtSF'] = [1 if i != 0 else 0 for i in Xtest['TotalBsmtSF']]
    # Xtest['Is_LowQualFinSF'] = [1 if i != 0 else 0 for i in Xtest['LowQualFinSF']]
    # Xtest['Is_GarageArea'] = [1 if i != 0 else 0 for i in Xtest['GarageArea']]
    # Xtest['Is_WoodDeckSF'] = [1 if i != 0 else 0 for i in Xtest['WoodDeckSF']]
    # Xtest['Is_OpenPorchSF'] = [1 if i != 0 else 0 for i in Xtest['OpenPorchSF']]
    # Xtest['Is_EnclosedPorch'] = [1 if i != 0 else 0 for i in Xtest['EnclosedPorch']]
    # Xtest['Is_3SsnPorch'] = [1 if i != 0 else 0 for i in Xtest['3SsnPorch']]
    # Xtest['Is_ScreenPorch'] = [1 if i != 0 else 0 for i in Xtest['ScreenPorch']]

    Xtest = ct.transform(Xtest)
    pred_ridge = np.exp(model_ridge.predict(Xtest)) - 1.0
    pred_lasso = np.exp(model_lasso.predict(Xtest)) - 1.0
    pred_xgb = np.exp(model_xgb.predict(Xtest)) - 1.0
    pred_stack = np.exp(model_stack.predict(Xtest)) - 1.0
    pred_svr = np.exp(model_svr.predict(Xtest)) - 1.0
    pred_rf = np.exp(model_rf.predict(Xtest)) - 1.0
    pred_lgbm = np.exp(model_lgbm.predict(Xtest)) - 1.0                

    # pred_ridge = np.exp(ppl_ridge.predict(Xtest)) - 1.0
    # pred_lasso = np.exp(ppl_lasso.predict(Xtest)) - 1.0
    # pred_xgb = np.exp(ppl_xgb.predict(Xtest)) - 1.0
    # pred_stack = np.exp(ppl_stack.predict(Xtest)) - 1.0
    # pred_svr = np.exp(ppl_svr.predict(Xtest)) - 1.0
    # pred_rf = np.exp(ppl_rf.predict(Xtest)) - 1.0
    # pred_lgbm = np.exp(ppl_lgbm.predict(Xtest)) - 1.0                

    # pred_ridge = inv_boxcox1p(ppl_ridge.predict(Xtest), 0.15)
    # pred_lasso = inv_boxcox1p(ppl_lasso.predict(Xtest), 0.15)
    # pred_xgb = inv_boxcox1p(ppl_xgb.predict(Xtest), 0.15)
    # pred_rf = inv_boxcox1p(ppl_rf.predict(Xtest), 0.15)
    # pred_lgbm = inv_boxcox1p(ppl_lgbm.predict(Xtest), 0.15)
    # pred_svr = inv_boxcox1p(ppl_svr.predict(Xtest), 0.15)
    # pred_stack = inv_boxcox1p(ppl_stack.predict(Xtest), 0.15)

    pred = 0.5 * pred_stack + \
        0.1 * pred_lgbm + \
        0.15 * pred_ridge + \
        0.15 * pred_lasso + \
        0.1 * pred_xgb

    # pred = pred_stack
    # pred = pred_lasso * 0.7 + pred_xgb * 0.3

    out = pd.DataFrame({'Id':testId, 'SalePrice':pred})
    # q1 = out['SalePrice'].quantile(0.0045)
    # q2 = out['SalePrice'].quantile(0.99)    
    # out['SalePrice'] = out['SalePrice'].apply(lambda x: x if x > q1 else x * 0.77)
    # out['SalePrice'] = out['SalePrice'].apply(lambda x: x if x < q2 else x * 1.1)
    # out['SalePrice'] *= 1.001619
    out.to_csv("submission_final.csv", index=False)
    return out

out = predict()    
# outcomp = pd.read_csv("../data/sample_submission.csv")
outcomp = pd.read_csv("./submission_xgb.csv")
plt.plot(out['SalePrice'], outcomp['SalePrice'], "b,")

# 1. Use boxcox
# 2. Use stack
# 3. Remove outliers
# 4. A fine tuning of the XGB parameter isn't very promising


# Best: 0.12509
# Add 3 outliers: 0.12670

print("Done.")


# This one outlier: Id = 2550
# 948458.4173298003 0.12388
# 8.e5 0.12241
# 6.e5 0.12025
# 3.e5 0.11690
# 1.e5 0.11729

