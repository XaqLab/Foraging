# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: xaqlab2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Intro
#
# In this notebook, we will investigate what learning the generative model of the task could look like. In particular, we will frame learning as inference over possible schedules given observations. First, we will establish theoretical upper bounds on the inference problem from a statistical estimation point of view. Then, we will construct Bayes-optimal posteriors, which can serve as a ground truth when assessing sufficient statistics uncovered by agents trained on the task.

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from foraging import plotting, utils
from foraging.config.constants import BOX_COLORS, BOX_LABELS, SEED
from foraging.plotting import PALETTE, bp, format_yticks, per_block, titler

from foraging.plotting.beliefs import (
    plot_fisher_info,
    plot_optimal_fisher_uncertainty,
    plot_schedule_beliefs_in_block,
    plot_schedule_beliefs_mean_and_std_across_blocks,
)
from foraging.utils import INDEX, MIN_INDEX
from foraging.utils.beliefs import compute_posteriors, get_mean_beliefs, get_std_beliefs
from foraging.utils.data import (
    display_df,
    exclusion_criteria,
    filter_df,
    make_df,
    process_blocks,
)
from foraging.utils.models import GammaBoxBelief, IndependentBoxesBelief

pd.options.mode.copy_on_write = True

# constants
RNG = np.random.default_rng(SEED)
DATA_DIR = "../data"
EXPERIMENT_DIR = os.path.join(DATA_DIR, 'experiments')

# %% [markdown]
# # Load data
#
# The experiment data consists of multiple matfiles corresponding to the data for different subjects. Each matfile contains that subject's push, eye tracking (if available), and position (if available) data organized by blocks and sessions. Each block corresponds to a set of experiment parameters, notably the schedule of each box, the stimulus reliability kappa, and the stimulus type. A hierarchical overview of a given matfile is given in the `Angelaki Data Cleaning` notebook.

# %%
# %%capture --no-display
df = make_df(os.path.join(DATA_DIR, "experiments"))
df = exclusion_criteria(df, EXPERIMENT_DIR)
display_df(df, ["box", "push times", "reward outcomes"])

# %% [markdown]
# # Estimating the Schedules
#
# In theory, observations about reward carries information about the task parameters that generate them. For example, if the subject is able to perfectly observe when reward becomes available at each box, then they can average these reward times to get an estimate of the mean schedule. It is known that the sample mean of observations drawn from a Gaussian distribution is the most efficient estimator of the population mean, in the sense that the variance achieves the Cramer-Rao lower bound. We will show the same is true for the sample mean of observations drawn from a Gamma distribution, which is the distribution we use in the experiment to generate reward intervals.
#
# Now, in reality, there are conditions in the experiment where the subject cannot perfectly observe the reward time, which is the case when the color cue encoding reward times contains spatiotemporal noise. On the other end of the spectrum, opposite from perfect observations, are completely unreliable observations that offer no information about the reward times, and hence no information about the schedules that generated them; there is still hope in the form of censored observations made available when the subject decides to push at a box. Then, the subject observes whether reward was made available by the time they pushed, and they can use this observation to update their beliefs about the schedules. For example, if the subject waits a long time at a box they believe to be fast but still does not observe reward when they push, that is an indication that the box may be slower than anticipated, and vice versa if the subject only waits a short time at a box they thought was slow and is pleasantly surprised to receive reward. Both the perfect observations and the censored observations correspond to distinct but related likelihood functions, described below:
#
# *Perfect Observations*
#
# $$
# L(\lambda) = gamma.pdf(o)
# $$
#
# *Censored Observations*
#
# $$
# L(\lambda) = gamma.cdf(o)
# $$
#
# From these, we can derive lower bounds on how fast the uncertainty can shrink for the ideal observer. First, in order to do that, we need to derive the Fisher information, which we go into next.
#
# #### Fisher Information
#
# The Fisher information tells us how discriminable the experiment parameters, such as the schedules, of the experiment are from similar parameters-- the higher the Fisher information, the higher the theoretical limit on our knowledge or certainty about the parameters. This relationship is formally known as the Cramer-Rao lower bound, which states that the variance of any unbiased estimator can be no lower than the inverse Fisher information. For example, if the subject knew the reward times $\{t_r\}$ exactly, then they could average them to get an estimate of the mean schedule, whose variance would be minimal among all estimators and thus equal to the inverse Fisher information.
#
# Alas, in reality, the subject's observations take the form of 1) censored observations $(r_t \in \{0, 1\})$ ie. was reward available by the time they pushed at time $t$ 2) noisy color cues that encode the reward time $o(t_r)$. These observations carry less information about the schedule than direct observations of the reward time, so naturally the variance of the estimators using these observations will not be minimal. Nonetheless, it is useful to know an upper bound on the precision of the estimation problem. We derive the Fisher information of the mean schedule ($\lambda$) as a function of the wait time ($t$), and the shape parameter ($\alpha$), using only observations about reward.
#
# $$J(\lambda) = (\frac{t}{\lambda})^2 \frac{f^2(t|\lambda, \alpha)}{F(t|\lambda, \alpha)(1-F(t|\lambda, \alpha))}$$
#
# Here, we show the Fisher information as a function of wait time for different boxes, along with the optimal waiting times that maximize the Fisher information, under the gamma schedule. The exponential schedule will be treated separately in a supplementary notebook.

# %%
# Cramer-Rao lower bound for standard deviation as a function of number of observations
import numpy as np
import matplotlib.pyplot as plt

means = [7, 14, 21]
shape = 10
n_obs = np.arange(1, 51)  # 1 to 50 observations
colors = list(PALETTE.values())
fisher_vals = [0.1303, 0.0326, 0.0145]
plt.figure(figsize=(8, 5))
for i, mu in enumerate(means):
    fisher_info = shape / mu**2
    std_crlb = np.sqrt(1 / (n_obs * fisher_info))
    std_crlb_2 = np.sqrt(1/(n_obs * fisher_vals[i]))
    plt.plot(n_obs, std_crlb, label=f"mean schedule = {mu}", color = colors[i])
    plt.plot(n_obs, std_crlb_2, color = colors[i], linestyle = "--")

plt.xlabel("# pushes")
plt.ylabel("standard deviation")
plt.title("Uncertainty shrinking with number of pushes (Perfectly reliable color cues vs reward observations only)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bias
#
# It's perfectly possible that the subjects have a prior about the schedules that could be mismatched to the true schedules ie. assigning an identical uniform prior to each box necessarily induces a bias in the beginning of the inference problem. Now, asymptotically the posterior converges to a Gaussian centered on the MLE, with variance equal to 1/fisher information (Laplace approximation), but in the beginning a nontrivial bias may be present that influences behavior. It is also worth thinking about constrained computation-- if the agent prematurely stops updating their posterior due to computational budget, then they may also be biased in their belief about the schedule depending on the nature of their limited observations. Depending on the nature of the bias, the MSE may even be less than that of the unbiased estimator (bias-variance decomposition)

# %%
plot_fisher_info(
    np.linspace(1, 30, 100),
    schedules=[7, 14, 21],
    alpha=10,
    title="Fisher information for Gamma schedule",
)
plt.show();

# %% [markdown]
# The optimal wait times that maximize the Fisher information of each box occur at distinct timepoints ordered by mean schedules. Notice that the maximal Fisher information of the two slower boxes are close together in value compared to the fast box, suggesting that the fast box may be easiest to figure out first because it requires not only fewer but also shorter pushes to reach a certain precision. We show how the precision grows with more observations by showing how the standard deviation, the square root of the Cramer-Rao lower bound, decreases with the number of obervations.

# %%

plot_optimal_fisher_uncertainty(
    20,
    [7, 14, 21],
    alpha=10,
    title="Fisher-optimal uncertainty as a function of number of pushes",
)
plt.show()


# %% [markdown]
# To get an estimate of the schedule down to within ±2-3 of the true schedule, one immediately starts of with such a precision after one push at the fast box under the gamma schedule. After 15-20 pushes, the precision of all boxes goes down to at least ±2-3 of the true schedule.

# %% [markdown]
#  # Beliefs about schedule
#
# Now that we have a sense of the theoretical upper bound on learning the schedules, we will look at the beliefs that a subject performing exact Bayesian inference may form over the actual censored observations $\{r_t\}$. We will consider the posteriors of each box to be pairwise independent and with uniform prior starting each block. The likelihood function is merely the cumulative density function $F$ of the gamma distribution if reward is delivered, and one minus this function otherwise. It is defined below for notational convenience:
#
# $$L(r_t) = \begin{cases} F(t) & \text{for } r_t = 1 \\ 1-F(t) &  \text{for } r_t = 0 \end{cases}$$
#
# Here is an example block's belief trajectories.

# %%
# Create a posterior class that is independent for each box
class Posterior(IndependentBoxesBelief):
    def __init__(self, n_boxes: int, *args, **kwargs):
        super().__init__(n_boxes, GammaBoxBelief, *args, **kwargs)


schedule_candidates = np.arange(30) + 1
schedule_beliefs, _ = process_blocks(
    df, compute_posteriors, Posterior, schedules=schedule_candidates, shape=10
)

conds = dict(subject="viktor", session=20230807, block=5)
plot_schedule_beliefs_in_block(df, schedule_beliefs, conds)

# %%

plot_schedule_beliefs_mean_and_std_across_blocks(
    df, schedule_beliefs, x="n pushes", min_obs=20
)

# %%

beliefs = schedule_beliefs
x = "push times"


# Get beliefs of each block
def _inner(df: pd.DataFrame, index: tuple):
    df_block = df.loc[index]
    posteriors = np.array(beliefs[index].features)
    schedule_candidates = beliefs[index].support[0]

    # Get mean and std of beliefs
    mean = get_mean_beliefs(posteriors, schedule_candidates)
    std = get_std_beliefs(posteriors, schedule_candidates)
    x_vals = df_block[x].values
    box = df_block["box"].values
    box_rank = df_block["box rank"].values
    res = {
        (index + (i,)): [
            mean[i + 1, box_rank[i]],
            std[i + 1, box_rank[i]],
            x_vals[i],
            box[i],
            i,
        ]
        for i in range(len(mean) - 1)
    }

    # Add prior
    for i in df_block["box rank"].unique():
        res[(index + (-i,))] = [mean[0, i], std[0, i], 0, box[0], 0]
    return res


res, _ = process_blocks(df, _inner)
res = reduce(lambda x, y: {**x, **y}, list(res.values()), {})
df_beliefs = pd.DataFrame.from_dict(
    res,
    orient="index",
    columns=["mean of belief", "s.d. of belief", x, "box", "n pushes"],
)


# %%

# df_beliefs.index = pd.MultiIndex.from_tuples(df_beliefs.index, names = INDEX[:MIN_INDEX])
# df_beliefs.sort_index(inplace=True)
bp(sns.lineplot)(
    df_beliefs,
    x="push times",
    y="mean of belief",
    hue="box",
    palette=PALETTE,
    min_obs=20,
)

# %%
bp(sns.lineplot)(
    df_beliefs,
    x="push times",
    y="s.d. of belief",
    hue="box",
    palette=PALETTE,
    min_obs=20,
)

# %%
df_beliefs


# %%
def mean(df: pd.DataFrame, index: tuple, exclude_prior: bool = True):

    # Get block data
    df_block = filter_df(df, conds)
    n_boxes = df_block["n boxes"].values[0]
    push_times = df_block["push times"].values
    posteriors = np.array(beliefs[tuple(conds.values())].features)
    schedule_candidates = beliefs[tuple(conds.values())].support[0]
    schedules = np.sort(df_block["schedule"].unique())
    mean_across_pushes = get_mean_beliefs(posteriors, schedule_candidates)[1:]

    res = utils.beliefs.get_mean_beliefs(
        posteriors[index].probabilities(record="all"), posteriors[index].support()
    )
    if exclude_prior:
        return [x[1:] for x in res]
    return res


def std(df: pd.DataFrame, index: tuple, exclude_prior: bool = True):
    res = utils.beliefs.get_std_beliefs(
        posteriors[index].probabilities(record="all"), posteriors[index].support()
    )
    if exclude_prior:
        return [x[1:] for x in res]
    return res


supp = np.arange(1, 30)
posteriors, err_beliefs = utils.data.process_blocks(
    df, utils.beliefs.compute_posteriors, supp, use_tqdm=True
)

mean_schedule, err_mean = utils.data.process_blocks(df, mean)
uncertainty_schedule, err_uncertainty = utils.data.process_blocks(df, std)
utils.data.extend_df(df, mean_schedule, "mean schedule", by_box=True)
utils.data.extend_df(df, uncertainty_schedule, "uncertainty schedule", by_box=True)

# %% [markdown]
# Here are the beliefs from an example block. Feel free to change the `index` values to change the block to display. You can also change `x_col` parameter to `push times` or even `push # by box` which shows the beliefs as a function of experience at each box.

# %%
conds = dict(subject="viktor", session=20230811, block=3)
plotting.beliefs.schedule_beliefs_block(
    df, tuple(conds.values()), x="push times", fig_kwargs={"figsize": (30, 10)}
)

# %%
plotting.behavior.plot_block_events(
    df, index, fig_kwargs={"figsize": (30, 2.2)}, legend=False
)

# %% [markdown]
# Aggregating across blocks, these are the means and uncertainties for Viktor, aligned by experience at each box.

# %%
conds = {"subject": "viktor"}
df_monkey = utils.data.filter_df(df, conds=conds, attempt_index=False)

# Binning data with pd.cut
bin_edges = [0, 5, 10, 15, 20] + list(range(30, 120, 10))
df_monkey["push # bins"] = utils.data.bin_data(df_monkey, "push # by box", bin_edges)
ax = bp(sns.lineplot)(
    df_monkey,
    conds=conds,
    x="push # bins",
    y="mean schedule",
    accumulate=True,
    title_prefix="Mean belief as function of experience in block",
    min_obs=20,
)

# %%
ax = bp(sns.lineplot)(
    df_monkey,
    conds=conds,
    x="push # bins",
    y="mean schedule",
    accumulate=True,
    title_prefix="Mean belief as function of experience in block",
    min_obs=20,
)

# %%
bp(sns.lineplot)(
    df_monkey,
    conds=conds,
    x="push # bins",
    y="uncertainty schedule",
    accumulate=True,
    title_prefix="Uncertainty as function of experience in block",
    min_obs=20,
)

# %%
df_ = df_monkey.xs(("viktor", 10), level=("subject", "shape")).reset_index()
subset_data = df_[df_["kappa"].isin([0, 0.1])]
sns.lineplot(subset_data, x="push # bins", y="uncertainty schedule", hue="kappa")

# %% [markdown]
# ## Likelihood

# %%
pts = np.linspace(0.2, 40)
LL = np.zeros((len(schedule_candidates) * len(pts), 3))
shape = 10
obs_model = utils.models.GammaObservation(shape)
c = np.zeros(LL.shape[0])
cnt = 0
for i, latent in enumerate(schedule_candidates):
    for j, pt in enumerate(pts):
        LL[cnt] = (latent, pt, obs_model.probability((1, pt), latent))
        c[cnt] = pt
        cnt += 1
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.scatter(LL[:, 0], LL[:, 1], LL[:, 2], c=c)
ax.set_xlabel("schedule")
ax.set_ylabel("push interval")
ax.set_title("Likelihood of possible schedules \nafter observing reward")
# df_monkey['same-box push intervals'].min(), df_monkey['schedule'].max()

# %% [markdown]
# # Reward beliefs


# %%
def rew_belief(df: pd.DataFrame, index: tuple, box):
    res = reward_beliefs[index][:, box, 1].squeeze()
    return res


def rew_prob(df: pd.DataFrame, index: tuple, box):
    res = reward_probabilities[index][:, box].squeeze()
    return res


rew_beliefs_blocks = [
    utils.data.process_blocks(df_monkey, rew_belief, box) for box in range(3)
]
rew_probs_blocks = [
    utils.data.process_blocks(df_monkey, rew_prob, box) for box in range(3)
]
[
    utils.data.extend_df(
        df_monkey, rew_beliefs_blocks[box][0], f"reward beliefs box={box}", by_box=False
    )
    for box in range(3)
]
[
    utils.data.extend_df(
        df_monkey, rew_probs_blocks[box][0], f"reward probs box={box}", by_box=False
    )
    for box in range(3)
]

# %%
subject = "viktor"
plotting.beliefs.reward_beliefs3d(
    df_monkey,
    df_monkey[[f"reward beliefs box={x}" for x in range(3)]].values,
    title=f"Belief about reward availability \nat time of push for {subject}",
)

# %%
plotting.beliefs.reward_beliefs3d(
    df_monkey,
    df_monkey[[f"reward probs box={x}" for x in range(3)]].values,
    title=f"Exact reward probability \nat time of push for {subject}",
)

# %% [markdown]
# ## Predict pushed box

# %%
# marginal beliefs
marginal_score, marginal_rsq, marginal_score_pval, marginal_rsq_pval, marginal_mdl = (
    utils.beliefs.predict_pushed_box(
        df_monkey, [f"reward beliefs box={x}" for x in range(3)], perm_test=True
    )
)

# perfect beliefs
perfect_score, perfect_rsq, perfect_score_pval, perfect_rsq_pval, perfect_mdl = (
    utils.beliefs.predict_pushed_box(
        df_monkey, [f"reward probs box={x}" for x in range(3)], perm_test=True
    )
)
print(
    f"marginal belief results: \n\t(accuracy, p-value) = ({marginal_score}, {marginal_score_pval})\n\t(r-squared, p-value) = ({marginal_rsq},{marginal_rsq_pval})"
)
print(
    f"perfect model results: \n\t(accuracy, p-value) = ({perfect_score}, {perfect_score_pval})\n\t(r-squared, p-value) = ({perfect_rsq},{perfect_rsq_pval})"
)

# %%
min_obs = 20  # threshold for min obs for bin to be considered
belief_models = df_monkey.groupby("push # bins").apply(
    lambda g: (
        utils.beliefs.predict_pushed_box(
            g,
            [f"reward beliefs box={x}" for x in range(3)],
            perm_test=True,
            weight=True,
        )
        if len(g) >= min_obs
        else None
    )
)
perfect_models = df_monkey.groupby("push # bins").apply(
    lambda g: (
        utils.beliefs.predict_pushed_box(
            g, [f"reward probs box={x}" for x in range(3)], perm_test=True, weight=True
        )
        if len(g) >= min_obs
        else None
    )
)

# Accessing results for each group
beliefs_results = {}
for group, result in belief_models.items():
    print(f"\nGroup: {group}")
    if result is None:
        print("skipped!")
        continue
    print("result", result)
    beliefs_results[group] = result

# %%
perfect_results = {}
for group, result in perfect_models.items():
    print(f"\nGroup: {group}")
    if result is None:
        print("skipped!")
        continue
    print("result", result)
    perfect_results[group] = result

# %%
fig = plt.figure()
beliefs_arr = np.array([x[:-1] for x in beliefs_results.values()])
perfect_arr = np.array([x[:-1] for x in perfect_results.values()])
plt.plot(beliefs_results.keys(), beliefs_arr[:, 0], label="bayesian model", c="b")
plt.plot(perfect_results.keys(), perfect_arr[:, 0], label="perfect model", c="r")

# Denote significant points
sig_idx = beliefs_arr[:, 2] < 0.05
plt.scatter(
    np.array(list(beliefs_results.keys()))[sig_idx],
    beliefs_arr[sig_idx, 0],
    c="b",
    s=50,
)
sig_idx = perfect_arr[:, 2] < 0.05
plt.scatter(
    np.array(list(perfect_results.keys()))[sig_idx],
    perfect_arr[sig_idx, 0],
    c="r",
    s=50,
)

sig_idx = beliefs_arr[:, 2] > 0.05
plt.scatter(
    np.array(list(beliefs_results.keys()))[sig_idx],
    beliefs_arr[sig_idx, 0],
    c="b",
    s=50,
    marker="x",
)
sig_idx = perfect_arr[:, 2] > 0.05
plt.scatter(
    np.array(list(perfect_results.keys()))[sig_idx],
    perfect_arr[sig_idx, 0],
    c="r",
    s=50,
    marker="x",
)

plt.xlabel("push # by box")
plt.ylabel("% boxes correctly predicted")
plt.legend()

# %%
fig = plt.figure()
beliefs_arr = np.array([x[:-1] for x in beliefs_results.values()])
perfect_arr = np.array([x[:-1] for x in perfect_results.values()])
plt.plot(beliefs_results.keys(), beliefs_arr[:, 1], label="bayesian model", c="b")
plt.plot(perfect_results.keys(), perfect_arr[:, 1], label="perfect model", c="r")

# Denote significant points
sig_idx = beliefs_arr[:, 3] < 0.05
plt.scatter(
    np.array(list(beliefs_results.keys()))[sig_idx],
    beliefs_arr[sig_idx, 1],
    c="b",
    s=50,
)
sig_idx = perfect_arr[:, 3] < 0.05
plt.scatter(
    np.array(list(perfect_results.keys()))[sig_idx],
    perfect_arr[sig_idx, 1],
    c="r",
    s=50,
)

sig_idx = beliefs_arr[:, 3] > 0.05
plt.scatter(
    np.array(list(beliefs_results.keys()))[sig_idx],
    beliefs_arr[sig_idx, 1],
    c="b",
    s=50,
    marker="x",
)
sig_idx = perfect_arr[:, 3] > 0.05
plt.scatter(
    np.array(list(perfect_results.keys()))[sig_idx],
    perfect_arr[sig_idx, 1],
    c="r",
    s=50,
    marker="x",
)

plt.xlabel("push # by box")
plt.ylabel("pseudo r-squared")
plt.legend()


# %% [markdown]
# ## Likelihood under probabilistic policy with negligible action costs


# %%
def exclude_column_indices(n_cols: int, exclude_indices: np.ndarray) -> np.ndarray:
    """
    Generate a matrix where each row contains all column indices except for the one given in exclude_indices.

    Args:
        n_cols (int): Total number of columns.
        exclude_indices (np.ndarray): Array of column indices to exclude per row.

    Returns:
        np.ndarray: Matrix where each row contains all column indices except the excluded one.
    """
    all_indices = np.arange(n_cols)  # Create an array of all column indices
    return np.array([np.delete(all_indices, i) for i in exclude_indices])


def belief_likelihood_push(df: pd.DataFrame, index: tuple):
    df_block = df.loc[index]
    data = reward_beliefs[index]
    picked_probs = data[
        np.arange(data.shape[0])[:, None], df_block["box rank"].values[:, None], 1
    ].squeeze()
    not_picked_index = exclude_column_indices(3, df_block["box rank"].values)
    not_picked_probs = data[
        np.arange(data.shape[0])[:, None], not_picked_index, 1
    ].squeeze()
    # print(picked_probs.shape, not_picked_probs.shape, df_block['box rank'].values.shape, not_picked_index.shape)
    return picked_probs * (1 - not_picked_probs[:, 0]) * (1 - not_picked_probs[:, 1])


def rew_prob_likelihood_push(df: pd.DataFrame, index: tuple):
    df_block = df.loc[index]
    data = reward_probabilities[index]
    picked_probs = data[
        np.arange(data.shape[0])[:, None], df_block["box rank"].values[:, None]
    ].squeeze()
    not_picked_index = exclude_column_indices(3, df_block["box rank"].values)
    not_picked_probs = data[
        np.arange(data.shape[0])[:, None], not_picked_index
    ].squeeze()
    return picked_probs * (1 - not_picked_probs[:, 0]) * (1 - not_picked_probs[:, 1])


rew_beliefs_blocks, _ = utils.data.process_blocks(df_monkey, belief_likelihood_push)
rew_probs_blocks, _ = utils.data.process_blocks(df_monkey, rew_prob_likelihood_push)
utils.data.extend_df(df_monkey, rew_beliefs_blocks, f"likelihood beliefs", by_box=False)
utils.data.extend_df(
    df_monkey, rew_probs_blocks, f"likelihood reward probs", by_box=False
)

# %%
fig, ax = plt.subplots()
sns.lineplot(
    data=df_monkey,
    x="push # bins",
    y="likelihood beliefs",
    ax=ax,
    label="bayesian model",
)
sns.lineplot(
    data=df_monkey,
    x="push # bins",
    y="likelihood reward probs",
    ax=ax,
    label="perfect model",
)
ax.legend()
ax.set_ylabel("probabilistic policy likelihood")

# %%
df_monkey2 = utils.data.make_df(os.path.join(DATA_DIR, "experiments"))

# %%

data_pairs = defaultdict(lambda: np.zeros((3, 3)))
old_index = None
for index, block in utils.data.get_blocks(df_monkey2):
    box = block.loc[index + (1,), "box"].iloc[0]
    if old_index is None:
        old_index = index
        continue
    else:
        if (
            old_index[0] == index[0]
            and old_index[1] == index[1]
            and old_index[2] + 1 == index[2]
        ):
            _df = df_monkey2.loc[old_index]
            try:
                box_rank = _df.loc[_df["box"] == box, "box rank"].iloc[0]
                data_pairs[index[0]][box - 1, box_rank] += 1
            except:
                pass
        old_index = index

# %%
fig, ax = plt.subplots(4, 1, figsize=(10, 30))
for i, (subj, data) in enumerate(data_pairs.items()):
    sns.heatmap(data, annot=True, cmap="Blues", cbar=True, linewidths=0.5, ax=ax[i])
    # ax[i].imshow(data)
    ax[i].set_ylabel("physical box location of first push")
    ax[i].set_xlabel("speed of pushed box in previous block")
    ax[i].set_title(f"{subj}'s data")

# %% [markdown]
# # Thompson sampling

# %%

supp = np.linspace(0, 1, 50)
a, b = ts_posterior[("viktor", 30, 5)].probabilities(record="all")[0][25]
c, d = ts_posterior[("viktor", 30, 5)].probabilities(record="all")[1][25]
e, f = ts_posterior[("viktor", 30, 5)].probabilities(record="all")[2][25]
probs = [
    scipy.stats.beta.pdf(supp, a, b),
    scipy.stats.beta.pdf(supp, c, d),
    scipy.stats.beta.pdf(supp, e, f),
]
print(a, b, c, d, e, f)
print(
    len(ts_posterior[("viktor", 30, 8)].probabilities(record="all")[0]),
    len(ts_posterior[("viktor", 30, 8)].probabilities(record="all")[1]),
    len(ts_posterior[("viktor", 30, 8)].probabilities(record="all")[2]),
)
plt.plot(supp, probs[0], color=BOX_COLORS[0], label=BOX_LABELS[0])
plt.plot(supp, probs[1], color=BOX_COLORS[1], label=BOX_LABELS[1])
plt.plot(supp, probs[2], color=BOX_COLORS[2], label=BOX_LABELS[2])
plt.legend()
plt.xlabel("Probability of reward")
plt.title("Beta-distributed beliefs about reward probability at beginning of block")

# %%
LL_ts, err_ts = utils.data.process_blocks(df, utils.stats.ts_likelihood, ts_posterior)
