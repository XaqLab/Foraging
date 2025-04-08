import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

import utils.data
from . import data
from .data import process_block_safely

def mcfadden_pseudo_rsquared(mdl, X, y):
    # 2. **Log-Likelihood**: Compute log-likelihood using sklearn's model
    y_prob = mdl.predict_proba(X)  # Get predicted probabilities
    # Log-likelihood for the observed classes
    log_likelihood = np.sum(np.log(y_prob[np.arange(len(y)), y]))

    # 3. **Null Log-Likelihood**: Compute log-likelihood of the null model (intercept-only model)
    # For the null model, we predict the mean class probability for all observations
    mean_class_prob = np.mean(y_prob, axis=0)[y]
    null_log_likelihood = np.sum(np.log(mean_class_prob))
    return 1 - log_likelihood / null_log_likelihood


def permutation_test_logistic(X, y, mdl_accu, mdl_rsq, n_perms: int = 500, seed: int = 0, weights = None, disp: bool = False):
    rsq_vals = np.zeros(n_perms)
    accu_vals = np.zeros_like(rsq_vals)
    rng = np.random.default_rng(seed)
    for i in range(n_perms):
        y_shuffle = rng.permutation(y)
        # null_mdl = smf.mnlogit("y~X", {'y': y_shuffle, 'X': X}).fit(disp = disp)
        # yhat = np.argmax(np.atleast_2d(null_mdl.predict()), axis = 1)
        # accu_vals[i] = (yhat == y_shuffle).mean()
        # rsq_vals[i] = null_mdl.prsquared
        mdl = LogisticRegression()
        mdl.fit(X, y_shuffle, sample_weight=weights)
        accu_vals[i] = mdl.score(X, y_shuffle, sample_weight=weights)
        rsq_vals[i] = mcfadden_pseudo_rsquared(mdl, X, y_shuffle)
    return (rsq_vals >= mdl_rsq).sum() /  len(rsq_vals), (accu_vals >= mdl_accu).sum() /  len(accu_vals)


def kfold_sessions(df, k, test_size = 0.2, seed = 0):
    rng = np.random.default_rng(
        seed
    )
    for i in range(k):
        seed = rng.integers(0, 1e9)
        for subject in df.index.unique('subject'):
            for session in df.loc[subject].index.unique('session'):
                df_sess = df.xs((subject, session), level = ('subject', 'session'), drop_level=False)
                train_idx, test_idx = train_test_split(df_sess.index, test_size=test_size, random_state=seed)
                df.loc[train_idx, f'{i+1}-fold'] = 'train'
                df.loc[test_idx, f'{i+1}-fold'] = 'test'

def kfold_fit_eval(df, k, fit_func, eval_func, fit_kwargs = {}, eval_kwargs = {}, fit_name = ''):
    train_results = []
    test_results = []
    for i in range(k):
        df_train, df_test = df.loc[df[f'{i+1}-fold'] == 'train'].copy(), df.loc[df[f'{i+1}-fold'] == 'test'].copy()

        # Fit on df_train
        fit_model = fit_func(df_train, **fit_kwargs)

        # Get training results
        utils.data.extend_df(df_train, eval_func(df_train, fit_model, **eval_kwargs), f'{fit_name} train results')
        train_results.append(df_train)

        # Evaluate on df_test
        utils.data.extend_df(df_test, eval_func(df_test, fit_model, **eval_kwargs), f'{fit_name} test results')
        test_results.append(df_test)
    return pd.concat(train_results, axis = 0), pd.concat(test_results, axis = 0)


@process_block_safely
def null_likelihood(df: pd.DataFrame, index: tuple):
    return np.ones(len(df.loc[index])) * np.log(1/3)


@process_block_safely
def biased_coin_likelihood(df: pd.DataFrame, index: tuple, probs):
    df_block = df.loc[index]
    return np.log(probs.loc[[(index[:2] + (x,)) for x in df_block['box label']]].values.squeeze())


@process_block_safely
def stay_switch_likelihood(df: pd.DataFrame, index: tuple, probs):
    df_block = df.loc[index]
    LL = np.zeros(len(df_block))
    mask = df_block['prev_box1'].isna()
    df_block = df_block.dropna(subset = ['prev_box1'])
    LL[~mask] = np.log(probs.loc[[(index[:2] + (x,)) for x in df_block['prev_box1']]].values[
        range(len(df_block)), df_block['curr_box']])
    LL[mask] = np.nan
    LL[LL == -np.inf] = np.nan
    return LL


@process_block_safely
def mc1_likelihood(df: pd.DataFrame, index: tuple, transition_probs):
    df_block = df.loc[index]
    LL = np.zeros(len(df_block))
    probs = transition_probs.loc[index[:2]]
    for i in range(LL.shape[0]):
        x = df_block.iloc[i]
        if pd.notna(x['prev_box1']):
            LL[i] = np.log(probs[x['ro_1']].loc[(x['prev_box1'],), x['curr_box']])
        else:
            LL[i] = np.nan
    LL[LL == -np.inf] = np.nan
    return LL


@process_block_safely
def generalized_markov_likelihood(df: pd.DataFrame, index: tuple, transition_probs):
    df_block = df.loc[index]
    LL = np.zeros(len(df_block))
    probs = transition_probs.loc[index[:2]]
    for i in range(LL.shape[0]):
        x = df_block.iloc[i]
        if pd.notna(x['prev_box1']) and pd.notna(x['prev_box2']):
            key = (x['ro_1'], x['ro_2'])
            LL[i] = np.log(probs[key].loc[(x['prev_box1'], x['prev_box2']), x['curr_box']])
        else:
            LL[i] = np.nan
    LL[LL == -np.inf] = np.nan
    return LL


# todo: the normalization here is a shortcut, long-term we need to write a policy class capable of evaluating the likelihood of each action
# @process_block_safely
def beliefs_likelihood(df: pd.DataFrame, index: tuple, beliefs):
    df_block = df.loc[index]
    data = beliefs[index] / beliefs[index].sum(axis = 1, keepdims = True)
    LL = np.log(data[np.arange(data.shape[0])[:, None], df_block['box rank'].values[:,None], 1].squeeze())
    LL[LL == -np.inf] = np.nan
    return LL


def rew_prob_likelihood(df: pd.DataFrame, index: tuple, probs):
    df_block = df.loc[index]
    data = probs[index] / probs[index].sum(axis = 1, keepdims = True)
    LL = np.log(data[np.arange(data.shape[0])[:, None], df_block['box rank'].values[:,None]].squeeze())
    LL[LL == -np.inf] = np.nan
    return LL

def sensory_likelihood(df: pd.DataFrame, index: tuple, probs):
    df_block = df.loc[index]
    data = np.exp(-probs[index]) / np.exp(-probs[index]).sum(axis = 1, keepdims = True)
    LL = np.log(data[np.arange(data.shape[0])[:, None], df_block['box rank'].values[:,None]].squeeze())
    LL[LL == -np.inf] = np.nan
    return LL


def null_likelihood_bins(df, x: str = 'push times', bins: int = 20):
    df['bins'] = data.bin_data(df, x, bins)
    subjects = df.index.unique('subject')
    LL = {subject: [[]] * df.loc[(subject,),'bins'].nunique() for subject in subjects}
    n_obs_bins = {subject: np.zeros(df.loc[(subject,),'bins'].nunique()) for subject in subjects}
    bins2idx = {k: i for i, k in enumerate(sorted(df['bins'].unique()))}
    for index, g in df.groupby(['subject', 'session']):
        for block in g.index.unique('block'):
            df_block = df.loc[index + (block,)]
            for i, x in df_block.iterrows():
                bin = bins2idx[x['push # bins']]
                LL[index[0]][bin].append(np.log(1 / 3))
                n_obs_bins[index[0]][bin] += 1
    return LL, n_obs_bins, 'bins'


