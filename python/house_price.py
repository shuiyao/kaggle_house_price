import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        sns.distplot(train['SalePrice'])

def study_variable_correlations():
    # Always a good practise to check correlation
    # CAVEAT: Only applies to continuous data, ignore categorical data
    # Default: Pearson correlation (linear correlation)
    # Variants: Kendall/Spearman Rank correlation.
    corr_p = train.corrwith(train['SalePrice'])
    corr_k = train.corrwith(train['SalePrice'], method="kendall")
    corr_s = train.corrwith(train['SalePrice'], method="spearman")
    # Create pandas.DataFrame from pandas.Series
    corrs = {'pearson':corr_p, 'kendall':corr_k, 'spearman':corr_s}
    corrs = pd.DataFrame(corrs)
    # corrs = corrs.T.sort_values(by=0)
    corrs.sort_values(by='pearson', inplace=True)
    # Now move on to the correlation matrix
    corrmat = train[corrs.index].corr()
    # Trick 1: re-order the training data to follow the order of correlation
    # Trick 2: Use a diverging color map rather than the default
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(corrmat, vmax=0.8, square=True, cmap=cmap)
    # The squared == True is very useful
    return corrs

# Check if a feature has NaN
# e.g. train['PoolQC'].isna().any()
# Check the non-NaN values of a feature
# e.g. train['PoolQC'].dropna()
# e.g. train['PoolQC'].value_counts()

# Often we want to pick out features that are mostly NULL.
# (doubtful) We will discard them if they are not strong indicators of Y.
# Should we discard features that have many missing values? It depends. For example, considering some houses that used to be hosted by very famous people or have experienced some bad rumors (though this sample doesn't have these kinds of information), this single feature will likely dominate its saleprice even though most houses do not have these features. Ignoring these features may lead to drastic over or under estimate of the house price. 
def study_missing_data():
    # One way to look at missing features
    print((train.count() / m).sort_values()[:10])
    # I found this a very illustrative way of checking missing values
    # (House Prices: EDA, Pipelines, GridSearchCV)
    fig, axs = plt.subplots(2, 1, num="missing_data", figsize=(10, 10))
    sns.heatmap(train.isna(), yticklabels=False, cbar=False, ax=axs[0])
    axs[0].set_xticks([])
    # Note that the figure above hides the labels of some features. One small update from myself is to only display features that have some (say, >1%) missing data:
    features_with_missing_data = train.loc[:, train.count() < 0.99 * m]
    sns.heatmap(features_with_missing_data.isna(), yticklabels=False, cbar=False, ax=axs[1])
    fig.subplots_adjust(bottom=0.2, hspace=0.1)
    
    # Select features with > 80% NaN
    sub_train_miss = train.loc[:, train.count() < 0.2 * m]
    # Print out the values, are they important?
    for (col, val) in sub_train_miss.iteritems():
        print("------------")
        print(val.value_counts())
    # Study their correlation with Y

def study_categorical_features(corrs=None):
    # What features are likely to be categorical?
    # 1. Non-numerical values.
    # 2. Lots of duplicates
    # 3. Numerical values but limited to a small range [0, n)
    # Note on 2: In practice, if some features have only a few non-unique values, it might just be treated as categorical.

    # How to pick them out?
    # 1. The corr() function automatically finds numerical columns.
    if(corrs is not None):
        non_num_cols = train.drop(labels=corrs.index, axis=1)
    # 3. Find the number of unique values in any feature
    ratio_unique = train.nunique() / train.count()
    # sns.ecdfplot(ratio_unique)
    # The 10 features with the most unique values
    # pandas guide says nlargest() is faster than sort_values().head()
    print("Features with least fraction of unique values: ")
    print(ratio_unique.nsmallest(10))
    return non_num_cols

def barplot_categorical_variables(var):
    '''
    Function to show dependence of Y (SalePrice) on any of the categorical variable var. Left: sns.histplot. Right: sns.boxplot. Note: sns.displot is figure-level procedure and does not work with ax=ax, use histplot, kdeplot, ... instead.
    '''
    fig, axs = plt.subplots(1, 2, num="barplot_catvars", figsize=(8,4))
    pdis = sns.histplot(data=train, x=var, y="SalePrice", ax=axs[0])
    pdis.set(title="histplot")
    # pdis.set(xticks=range(var))
    ylabel = ['{:,.0f}'.format(y) + 'k' for y in pdis.get_yticks() / 1000]
    pdis.set_yticklabels(ylabel)
    pdis.axis(ymin=0, ymax=800000)
    pbox = sns.boxplot(data=train, x=var, y="SalePrice", ax=axs[1])
    pbox.set(ylabel=None, yticklabels=[], title="boxplot")
    pbox.axis(ymin=0, ymax=800000)

def boxplot_multi_variables(cols, height=2.5, max_features=16):
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
    fig, axs = plt.subplots(nrows, ncols, num="boxplot_multi", figsize=(ncols*height, nrows*height*1.2), sharey=True)
    for i, ax in enumerate(axs.flatten()):
        if(i < num_features):
            sns.boxplot(data=train, x=cols[i], y="SalePrice", ax=ax)
        if(i % nrows): ax.set_ylabel("")
    fig.subplots_adjust(hspace=0.2, wspace=0, top=0.9)
    return fig, axs

def pairplot_multi_variables(cols):
    sns.set(style="ticks", font_scale=0.6)
    p = sns.pairplot(train[cols], height=1.5, corner=True, plot_kws={"s":5})
    # can use kind="hist"
    sns.despine()
    # Set corner to True to reduce resource consumption
    # Re-set tick labels for SalePrice
    for ax in p.axes.reshape(-1):
        if(ax == None): continue
        if(ax.get_xlabel() == 'SalePrice'):
            xlabels = ['{:,.0f}'.format(x) + 'k' for x in ax.get_xticks()/1000]
            ax.set_xticklabels(xlabels)
        if(ax.get_ylabel() == 'SalePrice'):
            ylabels = ['{:,.0f}'.format(y) + 'k' for y in ax.get_yticks()/1000]
            ax.set_yticklabels(ylabels)
    return p

# Data Cleaning
# --------------------------------
# Common procedures in data cleaning:
# 1. Drop leaky data.
# 2. Remove outliers
# 3. Feature scaling
# 4. Projection (dimensionality reduction)
# 5. Encoding

# study_missing_data()

# barplot_categorical_variables("CentralAir")

# corrs = study_variable_correlations()

# cols = corrs.nlargest(6, columns='pearson')
# p = multi_variable_pair_plot(cols.index)

# catvars = study_categorical_features(corrs)

# Now plotting the features with many missing data.
# cols = ["Alley", "PoolQC", "Fence", "MiscFeature"]
# fig, axs = boxplot_multi_variables(cols)

# fig, axs = boxplot_multi_variables(catvars.columns[:16])



# Drop leaky data
# --------------------------------
# In this case, YrSold, MoSold, SaleType, SaleCondition could be leaky features if we want to predict the sale price for houses that have not been sold yet. The trained model incorporates these information but the test data (houses not yet sold) do not have them.
# In general, data leakage manifests in various ways and often leads to over-estimation of the expected model performance. Ref: [https://insidebigdata.com/2014/11/26/ask-data-scientist-data-leakage/]

cols_leaky = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']
fig, axs = boxplot_multi_variables(cols_leaky)

plt.show()


# THOUGHTS:
# 1. USE PCA to reduce dimension if two features are strongly correlated
# 2. How to use categorical data

print("Done.")
