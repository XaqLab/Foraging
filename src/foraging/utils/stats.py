import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
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
    mask = df_block['next box'].isna()
    df_block = df_block.dropna(subset = ['next box'])
    LL[~mask] = np.log(probs.loc[[(index[:2] + (x,)) for x in df_block['box rank']]].values[
        range(len(df_block)), df_block['next box']])
    LL[mask] = np.nan
    LL[LL == -np.inf] = np.nan
    return LL

@process_block_safely
def winstay_loseswitch_likelihood(df: pd.DataFrame, index: tuple, probs_win, probs_lose):
    df_block = df.loc[index]
    LL = np.zeros(len(df_block))
    probs = [probs_win.loc[index[:2]], probs_lose.loc[index[:2]]]
    for i in range(LL.shape[0]):
        x = df_block.iloc[i]
        if pd.notna(x['next box']):
            LL[i] = np.log(probs[int(x['reward outcomes'])].loc[(x['box rank'],), x['next box']])
        else:
            LL[i] = np.nan
    LL[LL == -np.inf] = np.nan
    return LL

# @process_block_safely
def beliefs_likelihood(df: pd.DataFrame, index: tuple, beliefs):
    df_block = df.loc[index]
    data = beliefs[index]
    LL = np.log(data[np.arange(data.shape[0])[:, None], df_block['box rank'].values[:,None], 1].squeeze())
    LL[LL == -np.inf] = np.nan
    return LL

def rew_prob_likelihood(df: pd.DataFrame, index: tuple, probs):
    df_block = df.loc[index]
    data = probs[index]
    LL = np.log(data[np.arange(data.shape[0])[:, None], df_block['box rank'].values[:,None]].squeeze())
    LL[LL == -np.inf] = np.nan
    return LL


def null_likelihood_bins(df, x: str = 'push times', bins: int = 20):
    df['bins'] = utils.data.bin_data(df, x, bins)
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


