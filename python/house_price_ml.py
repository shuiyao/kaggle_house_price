import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import uniform

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit, learning_curve, cross_val_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn import metrics

import joblib

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
                       train_sizes=np.linspace(0.1, 0.8, 8))
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

cols_leaky = ['Id', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

print("Loading data from train.csv.")
train = pd.read_csv("train.csv")
train.info()

print("Dropping leaky features.")
m = train.shape[0]
train.drop(cols_leaky, axis=1, inplace=True)

# Log transform
train['SalePrice'] = np.log1p(train['SalePrice'])

# Now separate the training data into training and test set.
# The "test set" here is actually a hold-out cross validation set used to test if our ML models perform well. Later, we split the "train set" into k-Fold train/cross validation sets in order to find the hyperparameters that optimize average scores on the train/cv split. Finally we will use the optimized model on the hold-out set, before we apply it to the "real" test set provided in the test.csv file.
print("Separating train and test (hold-out cross-validation) set.")
X_train, X_test, y_train, y_test = \
    train_test_split(train, train['SalePrice'], test_size=0.3, random_state=120)

X_train.drop('SalePrice', axis=1, inplace=True)
X_test.drop('SalePrice', axis=1, inplace=True)

# Caveat: 
# An error raises if one passes DataFrame to the ColumnTransformer:
# "No valid specification of the columns. Only a scalar, list or slice of all integers or all strings, or boolean mask is allowed"
# Instead, one should pass indices (DataFrame.columns)
num_features = X_train.select_dtypes(['int', 'float']).columns
cat_features = X_train.select_dtypes('object').columns

model_lr = LinearRegression() # Toy model
model_sgd = SGDRegressor() # Linear Regressor with Stochastic Gradient Descent
model_ridge = Ridge()
model_lasso = Lasso(alpha=1.0)
model_rf = RandomForestRegressor() # Random Forest Decision Tree
model_xgb = XGBRegressor() # Extreme Gradient Boosting Decision Tree
# VotingRegressor()
# StackRegressor()

# Note: Sklearn UG says SGD is more suitable for large data (m > 10,000), otherwise use Lasso or Ridge

# Let's try linear regression model first without using pipeline

# imp_num = SimpleImputer()
# imp_num.fit(num_features)
# imp_num.transform(num_features)
# imp_cat = SimpleImputer(strategy='constant', fill_value='zero')
# imp_cat.fit(cat_features)
# xcat = imp_cat.transform(cat_features)
# ohe = OneHotEncoder(handle_unknown = 'ignore')
# ohe.fit(xcat)
# x = ohe.transform(xcat)

print("Preparing the data for both numeric and categorical data.")
# Numeric features
# ----------------
# Imputer: SimpleImputer(missing_values=np.nan, strategy={"mean","median","constant"})
# Fill missing values with mean() or median() of the given columns, or constant value specified by fill_value
#
# Scaler: StandardScaler() standardizes continuous variable to (mean, std) = (0.,1.)

ppl_num = Pipeline(steps = \
    [('imp_num', SimpleImputer()), \
     ('scaler', StandardScaler())] \
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
    [('imp_cat', SimpleImputer(strategy='constant', fill_value='zero')), \
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
def train_models(models, show_params=True):
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
        grid_params = {'lasso__alpha':[0.3, 1.0, 3.0, 10., 30., 100., 300., 1000.]}
        reg_lasso = GridSearchCV(ppl_lasso, grid_params, \
                                 cv=5, verbose=3, n_jobs=-1, \
                                 scoring = 'r2', return_train_score=True) # r2
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
        return model, reg_lasso
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
def train_model_rf_parameters_search(auto=True, par=None):
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
        ppl_rf.set_params(rf__n_estimators = 400, rf__max_features='sqrt', rf__max_depth=None, rf__min_samples_split = 5, rf__min_samples_leaf = 1)

        params = {'rf__n_estimators':[50, 100, 200, 400, 800, 1600], \
                  'rf__max_features':['auto', 'sqrt', 'log2'], \
                  'rf__max_depth':[2, 5, 10, 20, 50, 100], \
                  'rf__min_samples_split':[2, 5, 10, 15, 20], \
                  'rf__min_samples_leaf':[1, 2, 5, 10, 15]}
        scores_train, scores_cv = [], []
        parlst = params["rf__"+par]
        for parval in parlst:
            print("-------- %s = %s --------" % (par, str(parval)))
            ppl_rf.set_params(**{"rf__"+par:parval})
            ppl_rf.fit(X_train, y_train)
            scores_train.append(ppl_rf.score(X_train, y_train))
            scores_cv.append(ppl_rf.score(X_test, y_test))
        plt.plot(parlst, scores_train, "b.-")
        plt.plot(parlst, scores_cv, "r.-")
        ax = plt.gca()
        ax.set_xlabel(par)
        ax.set_ylabel("r2 score")
        ax.set_xticks(params["rf__"+par])
        ax.set_xticklabels([parval for parval in parlst])
        return _, _

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
        ppl_xgb.set_params(xgb__n_estimators = 400, xgb__min_child_weight=1, xgb__max_depth=3, xgb__gamma = 0, xgb__learning_rate = 0.1, xgb__subsample = 1, xgb__reg_lambda=1.0)
        # Use the same train/cv split
        params = { \
                   'xgb__min_child_weight': [1, 2, 3, 5, 8, 10], \
                   'xgb__n_estimators': [20, 50, 100, 200, 400, 800], \
                   'xgb__max_depth': [2, 3, 5, 7, 9], \
                   'xgb__subsample': [0.3, 0.5, 0.7, 0.9], \
                   'xgb__gamma': [0.0, 0.1, 0.5, 1.0, 5.0], \
                   'xgb__learning_rate': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32], \
                   'xgb__reg_lambda': [0.1, 0.3, 1.0, 3.0, 10.0, 30.0], \
                   'xgb__reg_alpha': [0.0, 0.3, 1.0, 3.0]
        }
        cpad = sns.color_palette() # The color pad
        scores, stds = [], []
        parlst = params["xgb__"+par]
        print("Training: ")
        best_score = -1.0
        best_model = None
        for parval in parlst:
            ppl_xgb.set_params(**{"xgb__"+par:parval})
            cvs = cross_val_score(ppl_xgb, X_train, y_train, cv=5, scoring='r2')
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
# model, reg_lasso = train_model_lasso_parameters_search(auto=True)
# model, reg_ridge = train_model_ridge_parameters_search()

# model, reg_rf = train_model_rf_parameters_search(auto=True)
# model, reg_rf = train_model_rf_parameters_search(auto=False, par='max_depth')

# xgb_params = ['n_estimators', 'min_child_weight', 'max_depth', 'gamma', 'subsample', 'learning_rate', 'reg_lambda', 'reg_alpha']
# fig, axs = plt.subplots(2,4,figsize=(10,8))
# axs = axs.flatten()
# for i, par in enumerate(xgb_params):
#     ppl, reg = train_model_xgb_parameters_search(auto=False, par=par, ax=axs[i])
# plt.show()

# CV accuracy is more sensitive to n_estimators, max_depth, learning_rate, subsample, reg_lambda within the parameter range.

ppl_xgb = Pipeline(steps = [('ct', ct), ('xgb', model_xgb)])
ppl_xgb.set_params(xgb__n_estimators = 400, xgb__min_child_weight=3, xgb__max_depth=3, xgb__gamma = 0, xgb__learning_rate = 0.04, xgb__subsample = 0.5, xgb__reg_lambda=1.0)

ppl_xgb.fit(X_train, y_train)
y_pred = ppl_xgb.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(metrics.r2_score(y_test, y_pred))
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

print("Done.")
