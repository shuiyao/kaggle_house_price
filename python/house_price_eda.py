import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import skew, pearsonr

# Feature examinations
# ================================
if 'train' not in globals():
    print("Loading data from train.csv.")
    train = pd.read_csv("train.csv")
    train.info()
    m = train.shape[0]

# Always First: Understanding Y:
def study_y(plot=False):
    if(plot == True):
        sns.displot(train['SalePrice'])
    print("Statistics of the target variable: ")
    print("--------------------------------")
    print(train['SalePrice'].describe())
    print("Skewness: %f" % (skew(train['SalePrice'])))

def study_variable_correlations(data, limit_score=1.0):
    # Always a good practise to check correlation
    # CAVEAT: Only applies to continuous data, ignore categorical data
    # Default: Pearson correlation (linear correlation)
    #   - However, outliers can be damaging to Pearson values
    # Variants: Kendall/Spearman Rank correlation.
    #   - One can apply it to ordinal categorical data
    corr_p = data.corrwith(data['SalePrice'])
    corr_k = data.corrwith(data['SalePrice'], method="kendall")
    corr_s = data.corrwith(data['SalePrice'], method="spearman")
    # Create pandas.DataFrame from pandas.Series
    corrs = {'pearson':corr_p, 'kendall':corr_k, 'spearman':corr_s}
    corrs = pd.DataFrame(corrs)
    corrs = corrs[abs(corrs['spearman']) > limit_score]
    # corrs = corrs.T.sort_values(by=0)
    corrs.sort_values(by='spearman', inplace=True)
    # Now move on to the correlation matrix
    corrmat = data[corrs.index].corr()#(method='spearman')
    # Trick 1: re-order the training data to follow the order of correlation
    # Trick 2: Use a diverging color map rather than the default
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    fig, ax = plt.subplots(1,1,figsize=(8,8), num="heatmap")
    sns.heatmap(corrmat, vmax=0.8, square=True, cmap=cmap, ax=ax)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    # The squared == True is very useful
    return corrs

# Check if a feature has NaN
# e.g. train['PoolQC'].isna().any()
# Check the non-NaN values of a feature
# e.g. train['PoolQC'].dropna()
# e.g. train['PoolQC'].value_counts()

# Often we want to drop features that are mostly NULL.
# (doubtful) We will discard them if they are not strong indicators of Y.
# Should we discard features that have many missing values? It depends. For example, considering some houses that used to be hosted by very famous people or have experienced some bad rumors (though this sample doesn't have these kinds of information), this single feature will likely dominate its saleprice even though most houses do not have these features. Ignoring these features may lead to drastic over or under estimate of the house price. 
def study_missing_data(data):
    # One way to look at missing features
    print((data.count() / m).sort_values()[:10])
    # I found this a very illustrative way of checking missing values
    # (House Prices: EDA, Pipelines, GridSearchCV)
    fig, axs = plt.subplots(2, 1, num="missing_data", figsize=(10, 10))
    sns.heatmap(data.isna(), yticklabels=False, cbar=False, ax=axs[0])
    axs[0].set_xticks([])
    # Note that the figure above hides the labels of some features. One small update from myself is to only display features that have some (say, >1%) missing data:
    features_with_missing_data = data.loc[:, data.count() < 0.99 * m]
    sns.heatmap(features_with_missing_data.isna(), yticklabels=False, cbar=False, ax=axs[1])
    fig.subplots_adjust(bottom=0.2, hspace=0.1)
    
    # Select features with > 80% NaN
    sub_data_miss = data.loc[:, data.count() < 0.2 * m]
    # Print out the values, are they important?
    for (col, val) in sub_data_miss.iteritems():
        print("------------")
        print(val.value_counts())
    # Study their correlation with Y
    return sub_data_miss

def study_categorical_features(corrs=None):
    # What features are likely to be categorical?
    # 1. Non-numerical values.
    # 2. Lots of duplicates
    # 3. Numerical values but limited to a small range [0, n)
    # Note on 2: In practice, if some features have only a few non-unique values, it might just be treated as categorical.

    # How to pick them out?
    # 1. The corr() function automatically finds numerical columns.
    # if(corrs is not None):
    #     non_num_cols = train.drop(labels=corrs.index, axis=1)
    vars_cat = train.select_dtypes('object')
    # 3. Find the number of unique values in any feature (cardinality)
    ratio_unique = vars_cat.nunique() / vars_cat.count()
    count_unique = vars_cat.nunique()
    # sns.ecdfplot(ratio_unique)
    # The 10 features with the most unique values
    # pandas guide says nlargest() is faster than sort_values().head()
    # One may want to avoid OneHot for high cardinality columns
    print("Features with least fraction of unique values: ")
    print(count_unique.nsmallest(10))
    return vars_cat

def barplot_categorical_variables(var):
    '''
    Function to show dependence of Y (SalePrice) on any of the categorical variable var. Left: sns.histplot. Right: sns.boxplot. Note: sns.displot is figure-level procedure and does not work with ax=ax, use histplot, kdeplot, ... instead.
    '''
    fig, axs = plt.subplots(1, 2, num="barplot_catvars", figsize=(8,8))
    pdis = sns.histplot(data=train, y=var, x="SalePrice", ax=axs[0])
    pdis.set(title="histplot")
    # pdis.set(xticks=range(var))
    xlabel = ['{:,.0f}'.format(x) + 'k' for x in pdis.get_xticks() / 2000]
    pdis.set_xticklabels(xlabel)
    pdis.axis(xmin=0, xmax=800000)
    # ---
    pbox = sns.boxplot(data=train, y=var, x="SalePrice", ax=axs[1], orient='h')
    pbox.set(ylabel=None, title="boxplot")
    xlabel = ['{:,.0f}'.format(x) + 'k' for x in pbox.get_xticks() / 2000]
    pbox.set_xticklabels(xlabel)
    pbox.axis(xmin=0, xmax=800000)
    fig.subplots_adjust(wspace=0.3)

def boxplot_multi_variables(cols, height=2.5, max_features=16, fignum="boxplot_multi", rank=False):
    '''
    Show the dependence of Y on multiple variables.
    upper: The maximum number of features to show (default: 4x4).
    '''
    num_features = len(cols)
    if (num_features > max_features):
        raise ValueError("Too many features (%d, limit=%d) to show." % (num_features, max_features))
    # Optimize the number of rows/cols for plotting
    nrows = int(np.sqrt(num_features))
    if(nrows * (nrows + 1) < num_features): nrows += 1
    ncols = int(np.ceil(num_features / nrows))
    print("Plotting %d x %d grid for %d features ..." % (nrows, ncols, num_features))
    fig, axs = plt.subplots(nrows, ncols, num=fignum, figsize=(ncols*height, nrows*height*1.2), sharey=True)
    for i, ax in enumerate(axs.flatten()):
        if(i < num_features):
            if(rank):
                xy = train[[cols[i],'SalePrice']]
                sorted_index = xy.groupby(cols[i]).mean().sort_values(by='SalePrice',ascending=False).index
                sns.boxplot(data=train, x=cols[i], y="SalePrice", ax=ax, order=sorted_index)
            else:
                sns.boxplot(data=train, x=cols[i], y="SalePrice", ax=ax)
        if(i % nrows): ax.set_ylabel("")
        xlabel = ax.get_xlabel()
        ax.text(0.95, 0.9, xlabel, transform=ax.transAxes, ha='right')
        ax.set_xlabel("")
    fig.subplots_adjust(hspace=0.2, wspace=0, top=0.9)
    return fig, axs

def pairplot_multi_variables(data, cols):
    sns.set(style="ticks", font_scale=0.6)
    p = sns.pairplot(data[cols], height=1.5, corner=True, plot_kws={"s":5})
    # can use kind="hist"
    sns.despine()
    # Set corner to True to reduce resource consumption
    # Re-set tick labels for SalePrice
    for ax in p.axes.reshape(-1):
        if(ax == None): continue
        if(ax.get_xlabel() == 'SalePrice'):
            ticks_loc = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
            xlabels = ['{:,.0f}'.format(x/1000) + 'k' for x in ticks_loc]
            ax.set_xticklabels(xlabels)
        if(ax.get_ylabel() == 'SalePrice'):
            ticks_loc = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
            ylabels = ['{:,.0f}'.format(y/1000) + 'k' for y in ticks_loc]
            ax.set_yticklabels(ylabels)        
    return p

# Data Cleaning and Feature Selection
# --------------------------------
# Why feature selection?
# 

# Common procedures in data cleaning:
# 1. Drop leaky data.
# 2. Feature selection (dimensionality reduction)
# 3. Remove outliers
# 4. Feature scaling
# 5. Encoding

# vars_sparse = study_missing_data()

# # Check if SalePrice is strongly correlated with these sparse features
# fig, axs = boxplot_mutli_variables(vars_sparse.columns[:], fignum="sparse_variables")

# barplot_categorical_variables("CentralAir")

# corrs = study_variable_correlations(train, limit_score=0.3)

# p = pairplot_multi_variables(['GarageYrBlt', 'YearBuilt', 'SalePrice'])
# p = pairplot_multi_variables(['GarageCars', 'GarageArea', 'SalePrice'])
# p = pairplot_multi_variables(['BsmtFinSF1', 'BsmtFinType1', 'SalePrice'])
# p = pairplot_multi_variables(['TotRmsAbvGrd', 'GrLivArea', 'SalePrice'])

# From the heatmap it is easy to pick out features that are strongly correlated:
# - The quality metrics (OverallQual, ExterQual, KitchenQual, BsmtQual, HeatingQC)
# - TotRmsAbvGrd and GrLivArea
# - BsmtFinSF1 and BsmtFinType1
# - GarageArea and GarageCars
# - YearBuilt and GarageYrBlt

# Using scatter plots help examine the correlated features. For example, for most houses the garage was built in the same year as the house, but there are exceptions where the garage was built later. It is unclear whether or not throwing away one of the features is a good idea. 

# Here is a nice explanation on multicollinearity in regression analysis.
# Ref: [https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/]
vars_collinear = ["GarageYrBlt","TotRmsAbvGrd","GarageCars", "BsmtFinSF1"]

#catvars = study_categorical_features()

# Now plotting the features with many missing data.
# cols = ["Alley", "PoolQC", "Fence", "MiscFeature"]
# fig, axs = boxplot_multi_variables(cols)

#fig, axs = boxplot_multi_variables(catvars.columns[:16])

# Drop leaky data
# --------------------------------
# In this case, YrSold, MoSold, SaleType, SaleCondition could be leaky features if we want to predict the sale price for houses that have not been sold yet. The trained model incorporates these information but the test data (houses not yet sold) do not have them.
# In general, data leakage manifests in various ways and often leads to over-estimation of the expected model performance. Ref: [https://insidebigdata.com/2014/11/26/ask-data-scientist-data-leakage/]

vars_leaky = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']
# fig, axs = boxplot_multi_variables(cols_leaky, fignum="leaky_variables")

# Scatter plot for features that are strongly correlated with SalePrice. Identify outliers.
# Do it after data cleaning.
# cols = corrs.nlargest(6, columns='pearson')
# p = pairplot_multi_variables(cols.index)

#plt.show()

# Now let's evaluate feature importance by fitting a random forest model.

from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

reg_rf = RandomForestRegressor()

import json
# Drop features that will not be used in modeling
# Combines features that would be dropped:
drop_features = ["Id"]
drop_features += vars_sparse.columns.tolist()
drop_features += vars_leaky
drop_features += vars_collinear
for var in drop_features:
    if(var in train):
        train = train.drop(var, axis=1)

# Hard code the ordinal features using a .json file
with open('ordinal.json',) as f:
    ordinal_feature_encoder = json.load(f)

train_trans = train.replace(ordinal_feature_encoder)
# MSSubClass is a nominal (categorical) feature but is encoded with numerics
train_trans['MSSubClass'] = train_trans['MSSubClass'].astype('object')

# Prepare data for training
y = train_trans['SalePrice']
X = train_trans.drop('SalePrice', axis=1)

num_features = X.select_dtypes(['int', 'float']).columns
cat_features = X.select_dtypes('object').columns
ord_features = pd.Index(ordinal_feature_encoder.keys()).join(X.columns, how='inner') # Note that some features have been dropped in X.

ppl_num = Pipeline(steps = \
    [('imp_num', SimpleImputer()), \
     ('scaler', StandardScaler())] \
)
ppl_cat = Pipeline(steps = \
    [('imp_cat', SimpleImputer(strategy='constant', fill_value='missing')), \
     ('ohe', OneHotEncoder(handle_unknown='ignore'))] \
)
# Must set handle_unknown, otherwise when applying to test data error may occur:
# "found unknown categories in column ... during transform"
ct = ColumnTransformer(transformers= \
    [('numeric', ppl_num, num_features), \
     ('categorical', ppl_cat, cat_features)] \
)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=120)

ppl_rf = Pipeline(steps = [('ct', ct), ('rf', reg_rf)])
ppl_rf.fit(X_train, y_train)
# First, check score to see if the fitted model is reasonable
print("r2 score (RandomForest): %f" % (ppl_rf.score(X_test, y_test)))

print("Analyzing feature importance using permutation method: ")
# Now how to extract feature importance from the fitted model? In simpler applications, the model has attributes such as coef_ and feature_importance_ that directly estimates the importance of any given feature. However there are two concerns: 1. The feature_importance_ are often estimated from the training data where the model may overfit. 2. It is hard to map the feature_importance_ metric to the original features if the data has gone through some transformations such as the OneHotEncoder for categorical features, which expands the feature space.
# permuation_importance() provides a very convenient way of evaluating the feature importance on a hold-out test set. However, the results may be degraded if many features are correlated.
perm_result = permutation_importance(ppl_rf, X_test, y_test, n_repeats=10, random_state=24)
perm_sorted_idx = perm_result.importances_mean.argsort()[::-1]

def show_important_features(perm_result, X):
    '''
    Display the most important features calculated from the permutation method.

    Parameters:
    ---
    perm_result: The output from sklearn.inspection.permutation_importance
    X: The training/test data before applying any pipeline. It is used to extract the features.
    '''
    types = ['N/A'] * X.columns.size
    feature_importance = pd.DataFrame({'mean':perm_result.importances_mean, \
                                       'std':perm_result.importances_std, \
                                       'type':types}, \
                                      index=X.columns)
    feature_importance.loc[num_features, 'type'] = 'numerical'
    feature_importance.loc[cat_features, 'type'] = 'nominal'
    feature_importance.loc[ord_features, 'type'] = 'ordinal'
    feature_importance.sort_values(by='mean', ascending=False, inplace=True)

    print("The top 10 Numerical features:")
    print(feature_importance[feature_importance['type']=='numerical'][:10])
    print("The top 5 Nominal features (nominal):")
    print(feature_importance[feature_importance['type']=='nominal'][:5])
    print("The top 5 Nominal features (ordinal):")
    print(feature_importance[feature_importance['type']=='ordinal'][:5])

    fig, ax = plt.subplots(1, 1, figsize=(8,6), num='feature_importance')
    sns.barplot(x = feature_importance['mean'][:20], \
                y = feature_importance.index[:20], \
                hue=feature_importance['type'][:20], \
                ax=ax, orient='h', dodge=False)
    ax.set_xscale("log")
    fig.subplots_adjust(left=0.2)
    plt.show()

show_important_features(perm_result, X)
print("Done.")
