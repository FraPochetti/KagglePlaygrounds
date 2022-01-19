def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import random
import os
import time
from scipy import stats
from scipy.stats import randint as sp_randint, uniform
import numpy as np
from joblib import dump, load
from IPython.core.display import display, HTML
import itertools
from pathlib import Path
from pandas.api.types import is_string_dtype, is_numeric_dtype
import re
from typing import List
# import pyodbc
from math import sqrt

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, KBinsDiscretizer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix, classification_report, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import calibration_curve
from sklearn.ensemble import IsolationForest
import xgboost as xgb

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import matplotlib

# from pdpbox import pdp
# import waterfall_chart

# import eli5
# from eli5.sklearn import PermutationImportance

# import lime
# import lime.lime_tabular

# from catboost import Pool, CatBoostClassifier, CatBoostRegressor
# import shap

#######################################################################

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def train_model(X_train, y_train, X_valid, y_valid, m=xgb.XGBClassifier(learning_rate=0.03, n_estimators=300, n_jobs=-1, verbosity = 0)):
    m.fit(X_train, y_train)
    probs_valid = m.predict_proba(X_valid)[:,1]
    return roc_auc_score(y_valid, probs_valid)

def estimate_valid_size_df(X, y, grid=np.arange(0.1, 1.1, 0.1), reps=range(30), verbose=False):
    valid_aucs = []

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=123)
    if verbose: print(f"Training on fixed {len(X_train)} points (70% total). Max validation size (30% total): {len(X_valid)}")

    m=xgb.XGBClassifier(learning_rate=0.03, n_estimators=300, n_jobs=-1, verbosity = 0)
    m.fit(X_train, y_train)
    probs_valid = m.predict_proba(X_valid)[:,1]
    valid = pd.DataFrame({'actual': y_valid, 'pred': probs_valid})

    for perc in grid:
        n = int(len(X_valid)*perc)
        if perc==1.0:
            auc = roc_auc_score(y_valid, probs_valid)
            valid_aucs.append((perc, n, auc, len(X_valid), len(X_train), 1))

        if perc<1.0:
            for _ in reps:
                val = valid.sample(n, replace=True)
                auc = roc_auc_score(val.actual, val.pred)
                valid_aucs.append((perc, n, auc, len(val), len(X_train), len(reps)))
    
    df = pd.DataFrame(valid_aucs, columns=['Percentage', 'Sample', 'AUC', 'Valid_size', 'Train_size', 'Bootstraps'])
    return df

def estimate_train_size_df(X, y, grid=np.arange(0.1, 1.1, 0.1), reps=range(30), verbose=False):
    since = time.time()
    train_aucs = []

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)
    if verbose: print(f"Validating on fixed {len(X_valid)} points (20% total). Max training size (80% total): {len(X_train)}")

    for perc in grid:
        n = int(len(X_train)*perc)
        if perc==1.0:
            auc = train_model(X_train, y_train, X_valid, y_valid)
            train_aucs.append((perc, n, auc, len(X_valid), len(X_train), 1))
            if verbose: print(f"Training once on {n} data points: {perc*100}% of {len(X_train)}...")
        
        if perc<1.0:
            if verbose: print(f"Training {len(reps)} times on {n} data points: {np.round(perc*100,1)}% of {len(X_train)}...")
            for _ in reps:
                X_t = X_train.sample(n)
                y_t = y_train.loc[X_t.index]
                auc = train_model(X_t, y_t, X_valid, y_valid)
                train_aucs.append((perc, n, auc, len(X_valid), len(X_t), len(reps)))
    time_elapsed = (time.time() - since)
    df = pd.DataFrame(train_aucs, columns=['Percentage', 'Sample', 'AUC', 'Valid_size', 'Train_size', 'Bootstraps'])
    print("Done in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    return df

def aggregate_size_df(df):
    df["Perc-Sample"] = (df.Percentage*100).astype(int).astype(str) + "%-" + df.Sample.astype(str)
    df = df.groupby('Perc-Sample').agg(Sample=('Sample', 'min'),
                                    Valid_size=('Valid_size','min'),
                                    Train_size=('Train_size','min'),
                                    Bootstraps=('Bootstraps', 'min'),
                                    AUC_mean=('AUC', 'mean'),
                                    AUC_std=('AUC', 'std'),
                                    AUC_975=('AUC', percentile(97.5)),
                                    AUC_025=('AUC', percentile(2.5))
                                    )
    df["975VSmean_%"] = (df.AUC_975/df.AUC_mean-1) * 100
    df["025VSmean_%"] = (df.AUC_025/df.AUC_mean-1) * 100
    df.sort_values(by='Sample', inplace=True)
    return df

def plot_size_df(df, title=None, plot_std=False):
    _, _ = plt.subplots(figsize=(9, 7))
    plt.plot(df.index, df.AUC_mean, 'k', label="Mean AUC")
    if plot_std: plt.fill_between(df.index, df.AUC_mean - 2 * df.AUC_std, df.AUC_mean + 2 * df.AUC_std, color='b', alpha=0.2, label="2std (95%) AUC interval")
    plt.fill_between(df.index, df.AUC_025, df.AUC_975, color='g', alpha=0.2, label="2.5-97.5 (95%) AUC quantiles")
    plt.ylabel('AUC')
    plt.xlabel('%dataset - #samples')
    if title is not None: plt.title(title)

    for x,y in zip(df.index,df.AUC_mean):
        label = "{:.3f}".format(y)
        plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.legend(loc="lower right")
    plt.xticks(rotation=30)
    plt.show()
    display(df.round(3))

def estimate_impact_size(what: str,
                         X: pd.DataFrame,
                         y: pd.Series, 
                         grid: np.array = np.arange(0.1, 1.1, 0.1), 
                         reps: range = range(30), 
                         verbose: bool = False) -> (pd.DataFrame, pd.DataFrame):
    """
    Estimates the impact of the training set size on a fix-sized validation set.

    Parameters
    ----------
    what: str
        `train` or `test`. Whether to estimate the impact of the size of the 
        training or test set.
    
    X : pd.DataFrame
        The dataframe containing all our dataset. Ready to be fed to an estimator.

    y: pd.Series
        The ground truth labels

    grid: np.array (default=np.arange(0.1, 1.1, 0.1))
        Array of percentages of the validation set to explore.

    reps: range (default=range(30))
        Number of times the validation process is repeated at each percentage level.
        Bootstrapping with repetition.
    
    verbose: bool (default=False)
        Whether to print relevant info while running
    """
    if what == 'test': original = estimate_valid_size_df(X, y, grid=grid, reps=reps, verbose=verbose)
    elif what == 'train': original = estimate_train_size_df(X, y, grid=grid, reps=reps, verbose=verbose)
    else: raise ValueError(f"`what` accepts `test` or `train` only: {what} was provided instead.")

    df = aggregate_size_df(original)
    if what == 'test': title = f"AUC on validation set of increasing size (up to 30% total - {df.Valid_size.max()} points) \n at fixed training set size (@70% total - {df.Train_size.max()} points)"
    else: title = f"AUC on validation set (fixed @20% total - {df.Valid_size.min()} points) \n at increasing training set size (up to 80% total - {df.Train_size.max()} points)"
    
    plot_size_df(df, title)
    return df, original

########################################################################

random_state = 10

def get_shap_explainer(which, clf):
    print(f'Model used: {type(clf)}')
    if which=='tree': return shap.TreeExplainer(clf)
    else: 
        raise ValueError(f'`get_shap_explainer` only supports `tree` for the moment. `{which}` was provided')

def get_shap_values(explainer, X): return explainer.shap_values(X) # those are log-odds not probabilities!        
        
def get_shap_feat_importance(shap_values, X, features=None, cats=None):
    if features is None: return shap.summary_plot(shap_values, X)
    shap.summary_plot(shap_values, X, feature_names=features, plot_type='bar')
    return pd.DataFrame(shap_values, columns=features)

def sigmoid(x): return 1/(1+np.exp(-x))

def show_shap(expected_value, shap_values, X, id=None, matplotlib=True, cats=None, x=None, y=None, disp=None, link='identity', feats=None, clf=None):
    if x is not None: return shap.dependence_plot(x, shap_values, X, display_features=disp, interaction_index=y)
    if id==None: return shap.force_plot(expected_value, shap_values, X)
    
    print('Printing top/bottom 5 features by SHAP values')
    if clf is not None: 
        try: print(f'Predicted probability of event: {clf.predict_proba(X.loc[id].values).squeeze()[1]}')
        except: print(f'Predicted probability of event: {clf.predict_proba(X.loc[id]).squeeze()[1]}')
    s = pd.DataFrame(shap_values, index=X.index, columns=feats)
    s1 = X.loc[id].to_frame('Feat value')
    s2 = s.loc[id].to_frame('SHAP value').join(s1)
    if cats is not None:
        for k, v in cats.items():
            try: 
                s2.loc[k, 'Feat value'] = v[int(s2.loc[k]['Feat value'])]
            except:
                pass
    with pd.option_context("display.max_rows", 10): display(s2.sort_values(by='SHAP value', ascending=False))
    return shap.force_plot(expected_value, s.loc[id].values, X.loc[id], link=link , matplotlib=matplotlib)
    
def assert_shap_proba(clf, explainer, shap_values, X, id):
    s = pd.DataFrame(shap_values, index=X.index)
    log_odds = explainer.expected_value + s.loc[id].values.sum()
    proba = sigmoid(log_odds)
    pred = clf.predict_proba(X.loc[id].values)[1]
    assert np.allclose(proba, pred)
    return proba, pred

def id2class(exp, cats):
    d = {}
    for f in exp.feature.values:
        if f in cats.keys():
            idx = int(exp.loc[exp.feature==f, 'value'].values[0])
            d[f] = f'{f}: {cats[f][idx]}'
    return d

def explain_pred_contrib(id, clf, X, features, cats=None, waterfall={'rotation_value':60, 'threshold': None}):
    try: p = clf.predict_proba(X.loc[X.index==id])[:, 1]
    except: p = clf.predict_proba(X.loc[X.index==id].values)[:, 1]
    print(f'Prediction explanation for ID: {id}; Probability of event (y=1): {np.round(p[0], 3)}\nModel used: {type(clf)}')
    try: 
        df = eli5.show_prediction(clf, X.loc[id], show_feature_values=True, feature_names=features)
        exp = eli5.explain_prediction_df(clf, X.loc[id], feature_names=features)
    except: 
        df = eli5.show_prediction(clf, X.loc[id].values, show_feature_values=True, feature_names=features)
        exp = eli5.explain_prediction_df(clf, X.loc[id].values, feature_names=features)

    if cats is not None:
        c = id2class(exp, cats)
        for k, v in c.items():
            df.data = df.data.replace(k, v)

    if waterfall is not None:
        rot = waterfall['rotation_value']
        threshold = waterfall['threshold']
        waterfall_chart.plot(exp.feature, exp.weight, rotation_value=rot,
                             net_label="Final Score/Proba", other_label="Minor Features", 
                             formatting="{:,.2f}", threshold=threshold, 
                             Title='Waterfall of features contributions')        
    return df

def get_lime_explainer(X_train, cat_feat_map, feats, kernel_width=None):
    categories = {e:{"i":i,"classes":cat_feat_map[e]} for i, e in enumerate(feats) if e in cat_feat_map.keys()}
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                    feature_names=feats,
                                                    class_names=cat_feat_map['income'],
                                                    categorical_features=[v['i'] for k,v in categories.items()],
                                                    categorical_names={v['i']:v['classes'] for k,v in categories.items()},
                                                    kernel_width=kernel_width,
                                                    random_state=random_state)
    return explainer    

def measure_lime_accuracy(clf, explainer, X_valid, num_features=10):
    lime_expl = X_valid.apply(explainer.explain_instance, predict_fn=clf.predict_proba, num_features=num_features, axis=1)
    lime_pred = lime_expl.apply(lambda x: x.local_pred[0])
    return r2_score(clf.predict_proba(X_valid.values)[:,1], lime_pred)

def explain_pred_lime(idx, clf, explainer, X_valid, num_features=10):
    print(f'Model used: {type(clf)}')
    exp = explainer.explain_instance(X_valid.loc[idx].values, 
                                    clf.predict_proba, num_features=num_features)
    local = exp.local_pred[0]
    pred = exp.predict_proba[1]
    print(f'LIME local prediction: {local}; Model prediction: {pred}; R2 score: {exp.score}')
    exp.show_in_notebook(show_table=True, show_all=False)
    return exp

def get_styled_feat_importance(m, features, show_top=None):
    return eli5.show_weights(m, feature_names=features, top=show_top)

def get_permutation_imp(m, X, y, feats, random_state=random_state, scoring='roc_auc'):
    perm_train = PermutationImportance(m, random_state=random_state, scoring=scoring)
    _ = perm_train.fit(X, y)    
    all_feat_imp_df = eli5.explain_weights_df(perm_train, feature_names=feats)
    
    perm_train_feat_imp_df = pd.DataFrame(data=perm_train.results_, columns=feats)
    perm_train_feat_imp_df = perm_train_feat_imp_df[list(all_feat_imp_df.feature)]
    ax = perm_train_feat_imp_df.iloc[:,:15].boxplot(figsize=(9,7))
    ax.set(title='Permutation Importance Distributions (training data)', ylabel='Importance')
    plt.xticks(rotation=90)
    plt.show()
    display(all_feat_imp_df[:15])
    
    return all_feat_imp_df

def reduce_cardinality(df, f, threshold=0.95):
    if len(df[f].unique()) < 6: return df
    nc = df[f].value_counts().to_frame()
    nc = (nc/nc[f].sum())[f].cumsum().to_frame()
    to_keep = nc.loc[nc[f]<=threshold,].index
    df[f] = np.where(df[f].isin(to_keep), df[f], 'UNK')
    return df

def plot_pdp(m, X, features, feature, center=True, classes=None, percentile_range=None, plot_params=None):
    p = pdp.pdp_isolate(m, X, features, feature, n_jobs=-1, percentile_range=percentile_range)
    fig, axes = pdp.pdp_plot(p, feature, plot_lines=True, center=center, plot_pts_dist=True, plot_params=plot_params)
    if classes is not None: 
        _ = axes['pdp_ax']['_pdp_ax'].set_xticklabels(classes)
        _ = axes['pdp_ax']['_count_ax'].set_xticklabels(classes)
        _ = axes['pdp_ax']['_count_ax'].set_xlabel('')
        _ = axes['pdp_ax']['_count_ax'].set_title('')
        fig.autofmt_xdate()
    plt.show()

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix validation set',
                          cmap=plt.cm.Blues):
    """
    plot_confusion_matrix prints and plots the cm 
    confusion matrix received in input.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    ax.set_ylim([1.5,-0.5])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.grid(b=None)

def plot_roc(y_valid, probs_valid, auc):
    fpr, tpr, thresholds = roc_curve(y_valid, probs_valid)
    ax, fig = plt.subplots(figsize=(6,6))

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.3f)' % auc)
    plt.legend(loc="lower right")
    plt.show()
    
def plot_calibration(y_valid, prob_pos, name='Current Model'):
    plt.figure(figsize=(7, 7))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, prob_pos, n_bins=20)
    
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))
    
    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)
    
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    
    plt.tight_layout()
    plt.show()
    
def print_report(m, X_valid, y_valid, t=0.5, X_train=None, y_train=None, conf=True, roc=True, 
                 cal=True, vstrue=False, verbose=True, classes=['Ok', 'Default']):
    """
    print_report prints a comprehensive classification report
    on both validation and training set (if provided).
    The metrics returned are AUC, F1, Precision, Recall and 
    Confusion Matrix.
    Results are dependent on the probability threshold t 
    applied to individual predictions.
    """
    
    probs_valid = m.predict_proba(X_valid)[:,1]
    y_val_pred = adjusted_classes(probs_valid, t)
        
    if X_train is not None:
        probs_train = m.predict_proba(X_train)[:,1]
        y_train_pred = adjusted_classes(probs_train, t)
    
    res = [roc_auc_score(y_valid, probs_valid),
           f1_score(y_valid, y_val_pred),
           confusion_matrix(y_valid, y_val_pred)]
    result = f'AUC valid: {res[0]} \nF1 valid: {res[1]}'
    
    if X_train is not None:
        res += [roc_auc_score(y_train, probs_train),
                f1_score(y_train, y_train_pred)]
        result += f'\nAUC train: {res[3]} \nF1 train: {res[4]}\n'

    report = classification_report(y_valid, y_val_pred, output_dict=True)
    report['1']['auc'] = res[0]
    if X_train is not None: report['1']['train_dr'] = y_train.sum()/len(y_train)
    report['1']['valid_dr'] = y_valid.sum()/len(y_valid)
    if verbose:
        print(result)
        print(classification_report(y_valid, y_val_pred))
    if conf: plot_confusion_matrix(res[2], classes=classes)
    if roc: plot_roc(y_valid, probs_valid, res[0])
    if cal: plot_calibration(y_valid, probs_valid)
    if vstrue: plot_bins(y_valid, probs_valid)

    return report

def plot_bins(y_valid, probs_valid):    
    valid = np.hstack((y_valid[:, None], probs_valid[:, None]))
    valid = pd.DataFrame(valid, columns=["Actuals", "Predicted"])
    valid['bins'] = pd.qcut(valid['Predicted'], 10)
    g = valid.groupby(["bins"])["Actuals", "Predicted"].mean()
    g.plot(rot=45, title="Actuals VS Predicted PD Bins")
    plt.tight_layout()
    plt.show()
    
    import seaborn as sns
    #sns.set(style="whitegrid")   
    plt.subplots(figsize=(8, 5))
    sns.distplot(valid[valid.Actuals==0]["Predicted"], hist_kws=dict(linewidth=0, alpha=0.5), 
                 kde_kws=dict(linewidth=2))
    sns.distplot(valid[valid.Actuals==1]["Predicted"], hist_kws=dict(linewidth=0, alpha=0.5), 
                 kde_kws=dict(linewidth=2))
    plt.title('Divergence: Default/Ok customers', fontsize=18)

    plt.legend(['Ok', 'Default'], fontsize = 14)
    plt.xlabel('Probability of Default', fontsize = 16)
    plt.ylabel('Density', fontsize = 16)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    plt.show()

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df.T)

def add_datepart(df, fldname, set_as_index=False, time=False, sort=True, drop=True):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    """
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 
            'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    #df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if sort: df.sort_values(by=fldname, inplace=True)
    if set_as_index: df.set_index(fldname, inplace=True)
    if drop and not set_as_index: df.drop(fldname, axis=1, inplace=True)
        
def missing_values(data):
    total = data.isnull().sum().sort_values(ascending = False) 
    percent = (data.isnull().sum() / data.isnull().count() * 100 ).sort_values(ascending = False) 
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
    return df[~(df['Total'] == 0)] 

def train_cats(df):
    """Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.
    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category
    """
    cats={}
    for n,c in df.items():
        if is_string_dtype(c): 
            df[n] = c.astype('category').cat.as_ordered()
            cats[n] = list(df[n].cat.categories)
    return cats

            
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    """ proc_df takes a data frame df and splits off the response variable, and
    changes the df into an entirely numeric dataframe.
    Parameters:
    -----------
    df: The data frame you wish to process.
    y_fld: The name of the response variable
    skip_flds: A list of fields that dropped from df.
    ignore_flds: A list of fields that are ignored during processing.
    do_scale: Standardizes each column in df. Takes Boolean Values(True,False)
    na_dict: a dictionary of na columns to add. Na columns are also added if there
        are any missing values.
    preproc_fn: A function that gets applied to df.
    max_n_cat: The maximum number of categories to break into dummy values, instead
        of integer codes.
    subset: Takes a random subset of size subset from df.
    mapper: If do_scale is set as True, the mapper variable
        calculates the values used for scaling of variables during training time (mean and standard deviation).
    Returns:
    --------
    [x, y, nas, mapper(optional)]:
        x: x is the transformed version of df. x will not have the response variable
            and is entirely numeric.
        y: y is the response variable
        nas: returns a dictionary of which nas it created, and the associated median.
        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continuous
        variables which is then used for scaling of during test-time.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category { a : 1, b : 2}
    >>> x, y, nas = proc_df(df, 'col1')
    >>> x
       col2
    0     1
    1     2
    2     1
    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])
    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())])
    >>>round(fit_transform!(mapper, copy(data)), 2)
    8x4 Array{Float64,2}:
    1.0  0.0  0.0   0.21
    0.0  1.0  0.0   1.88
    0.0  1.0  0.0  -0.63
    0.0  0.0  1.0  -0.63
    1.0  0.0  0.0  -1.46
    0.0  1.0  0.0  -0.63
    1.0  0.0  0.0   1.04
    0.0  0.0  1.0   0.21
    """
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: pass
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: pass
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: pass
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=False)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: pass
    return res

def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.
    Parameters:
    -----------
    df: The data frame that will be changed.
    col: The column of data to fix by filling in missing data.
    name: The name of the new filled column in df.
    na_dict: A dictionary of values to create na's of and the value to insert. If
        name is not a key of na_dict the median will fill any missing data. Also
        if name is not a key of na_dict and there is no missing data in col, then
        no {name}_na column is not created.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col2'], 'col2', {})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1   500    2    True
    2     3    2   False
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            print(col, na_dict)
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    """ Changes the column col from a categorical type to it's integer codes.
    Parameters:
    -----------
    df: A pandas dataframe. df[name] will be filled with the integer codes from
        col.
    col: The column you wish to change into the categories.
    name: The column name you wish to insert into df. This column will hold the
        integer codes.
    max_n_cat: If col has more categories than max_n_cat it will not change the
        it to its integer codes. If max_n_cat is None, then col will always be
        converted.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category { a : 1, b : 2}
    >>> numericalize(df, df['col2'], 'col3', None)
       col1 col2 col3
    0     1    a    1
    1     2    b    2
    2     3    a    1
    """
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = col.cat.codes

def apply_cats(df, trn):
    """Changes any columns of strings in df into categorical variables using trn as
    a template for the category codes.
    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values. The category codes are determined by trn.
    trn: A pandas dataframe. When creating a category for df, it looks up the
        what the category's code were in trn and makes those the category codes
        for df.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category {a : 1, b : 2}
    >>> df2 = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['b', 'a', 'a']})
    >>> apply_cats(df2, df)
           col1 col2
        0     1    b
        1     2    a
        2     3    a
    now the type of col is category {a : 1, b : 2}
    """
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)