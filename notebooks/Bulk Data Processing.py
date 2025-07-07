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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bulk Data Processing
# This notebook is for processing and saving data across multiple subjects in bulk.

# %% [markdown]
# # Setup
# Run this first to import everything and set up notebook

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import logging
import os
import pickle

import numpy as np

from foraging import utils
from foraging.utils import beliefs, data
from foraging.utils.models import GammaScheduleBelief

# Filter out annoying matplotlib logs
mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)

EXPERIMENTS_DIR = "../data/experiments"
SAVE_DIR = "../data/analysis"
SEED = 42

# %%

schedules = np.arange(1, 31)
posterior = GammaScheduleBelief(shape=10, schedules=schedules)

# %%
posterior.update((True, 10))

# %%

# %% [markdown]
# # Load experiment data
# Load matlab datafiles into a pandas Dataframe

# %%
df = utils.data.make_df(EXPERIMENTS_DIR)
df = utils.data.exclusion_criteria(df)
df.head()

# %% [markdown]
# # Compute beliefs
# Compute the posterior for all blocks where the color cue was uninformative

# %%
supp = np.arange(1, 30)
posteriors, err_beliefs = utils.data.process_blocks(
    df, utils.beliefs.compute_posteriors, supp, use_tqdm=True
)

# Assume agent assumes a shape parameter
posteriors_shape1, err_beliefs_shape1 = utils.data.process_blocks(
    df, utils.beliefs.compute_posteriors, supp, shape=1, use_tqdm=True
)
posteriors_shape10, err_beliefs_shape10 = utils.data.process_blocks(
    df, utils.beliefs.compute_posteriors, supp, shape=10, use_tqdm=True
)

# %%
func = lambda df, index, p: utils.beliefs.compute_latent_beliefs_over_time(
    df, index, p[index], dt=0.5, padding_time=0.5
)
latent_beliefs_time, err_latent_time = utils.data.process_blocks(
    df, func, posteriors, use_tqdm=True
)
latent_beliefs_time_shape1, err_latent_time_shape1 = utils.data.process_blocks(
    df, func, posteriors_shape1, use_tqdm=True
)
latent_beliefs_time_shape10, err_latent_time_shape10 = utils.data.process_blocks(
    df, func, posteriors_shape10, use_tqdm=True
)

# %%
func = lambda df, index, p: utils.beliefs.compute_joint_beliefs(df, index, p[index])
joint_beliefs, err_joint = utils.data.process_blocks(
    df, func, posteriors, use_tqdm=True
)
joint_beliefs_shape1, err_joint_shape1 = utils.data.process_blocks(
    df, func, posteriors_shape1, use_tqdm=True
)
joint_beliefs_shape10, err_joint_shape10 = utils.data.process_blocks(
    df, func, posteriors_shape10, use_tqdm=True
)

# %%
func = lambda df, index, p: utils.beliefs.compute_reward_beliefs(df, index, p[index])
reward_beliefs, err_reward = utils.data.process_blocks(
    df, func, posteriors, use_tqdm=True
)
reward_beliefs_shape1, err_reward_shape1 = utils.data.process_blocks(
    df, func, posteriors_shape1, use_tqdm=True
)
reward_beliefs_shape10, err_reward_shape10 = utils.data.process_blocks(
    df, func, posteriors_shape10, use_tqdm=True
)

# %% [markdown]
#
# ## Perfect model
# Compute the exact reward probabilities under a perfect model of the boxes (one where the schedules are known exactly)

# %%
reward_probabilities, err_reward_probabilities = utils.data.process_blocks(
    df, utils.beliefs.compute_reward_probabilities, use_tqdm=True
)
reward_probabilities_shape1, err_reward_probabilities_shape1 = (
    utils.data.process_blocks(
        df, utils.beliefs.compute_reward_probabilities, shape=1, use_tqdm=True
    )
)
reward_probabilities_shape10, err_reward_probabilities_shape10 = (
    utils.data.process_blocks(
        df, utils.beliefs.compute_reward_probabilities, shape=10, use_tqdm=True
    )
)


# %% [markdown]
# ## Wrong model


# %%
def flatten_schedules(df, index, factor=0.5):
    schedules = sorted(df.loc[index, "schedule"].unique())
    if len(schedules) != 3:
        return [
            schedules[0] + factor * (schedules[1] - schedules[0]),
            schedules[1] - factor * (schedules[1] - schedules[0]),
        ]
    min_val, mid_val, max_val = schedules
    adjusted_min = min_val + factor * (mid_val - min_val)
    adjusted_max = max_val - factor * (max_val - mid_val)
    return [adjusted_min, mid_val, adjusted_max]


wrong_schedules, err_flatten = utils.data.process_blocks(
    df, flatten_schedules, factor=-0.25
)
reward_beliefs_wrong, err_reward_beliefs_wrong = utils.data.process_blocks(
    df,
    lambda df, index: utils.beliefs.compute_reward_probabilities(
        df, index, schedules=wrong_schedules[index]
    ),
    use_tqdm=True,
)
reward_beliefs_wrong_shape1, err_reward_beliefs_wrong_shape1 = (
    utils.data.process_blocks(
        df,
        lambda df, index: utils.beliefs.compute_reward_probabilities(
            df, index, schedules=wrong_schedules[index], shape=1
        ),
        use_tqdm=True,
    )
)
reward_beliefs_wrong_shape10, err_reward_beliefs_wrong_shape10 = (
    utils.data.process_blocks(
        df,
        lambda df, index: utils.beliefs.compute_reward_probabilities(
            df, index, schedules=wrong_schedules[index], shape=10
        ),
        use_tqdm=True,
    )
)

# %% [markdown]
# ### Thompson sampling

# %%
func = lambda df, index, p: utils.beliefs.compute_reward_beliefs(df, index, p[index])
ts_posterior, err_ts_posterior = utils.data.process_blocks(
    df, utils.beliefs.compute_beta_posteriors, use_tqdm=True
)
ts_reward_belief, err_ts_reward_belief = utils.data.process_blocks(
    df, func, ts_posterior, use_tqdm=True
)

# %% [markdown]
# # Scrambled Reward Outcomes

# %%
df["reward outcomes"] = np.random.choice([0, 1], size=len(df["reward outcomes"]))
reward_probabilities_scrambled, err_reward_probabilities_scrambled = (
    utils.data.process_blocks(
        df, utils.beliefs.compute_reward_probabilities, use_tqdm=True
    )
)
# # Assume agent assumes a shape parameter
# posteriors_shape1, err_beliefs_shape1 = utils.data.process_blocks(df, utils.beliefs.compute_posteriors, supp, shape = 1, use_tqdm=True)
# posteriors_shape10, err_beliefs_shape10 = utils.data.process_blocks(df, utils.beliefs.compute_posteriors, supp, shape = 10, use_tqdm=True)

# %% [markdown]
# # Information

# %%
surprise, err_surprise = utils.data.process_blocks(
    df, utils.beliefs.compute_surprise, use_tqdm=True
)
fisher_info, err_fisher = utils.data.process_blocks(
    df, utils.beliefs.compute_fisher, use_tqdm=True
)

# %%
deriv, err_deriv = utils.data.process_blocks(
    df, utils.beliefs.compute_deriv, use_tqdm=True
)

# %% [markdown]
# # Save data

# %%
with open(os.path.join(SAVE_DIR, "bulk_beliefs.pkl"), "wb") as f:
    pickle.dump(
        {
            "data": {
                "posteriors": posteriors,
                "posteriors shape 1": posteriors_shape1,
                "posteriors shape 10": posteriors_shape10,
                "latent_beliefs_over_time": latent_beliefs_time,
                "latent_beliefs_over_time shape 1": latent_beliefs_time_shape1,
                "latent_beliefs_over_time shape 10": latent_beliefs_time_shape10,
                "joint_beliefs": joint_beliefs,
                "joint_beliefs shape 1": joint_beliefs_shape1,
                "joint_beliefs shape 10": joint_beliefs_shape10,
                "reward_beliefs": reward_beliefs,
                "reward_beliefs shape 1": reward_beliefs_shape1,
                "reward_beliefs shape 10": reward_beliefs_shape10,
                "reward probabilities": reward_probabilities,
                "reward probabilities shape 1": reward_probabilities_shape1,
                "reward probabilities shape 10": reward_probabilities_shape10,
                "reward beliefs wrong": reward_beliefs_wrong,
                "reward beliefs wrong shape 1": reward_beliefs_wrong_shape1,
                "reward beliefs wrong shape 10": reward_beliefs_wrong_shape10,
                "surprise": surprise,
                "fisher": fisher_info,
                "deriv": deriv,
                "scrambled": reward_probabilities_scrambled,
            },
            "error": {
                "err_schedule_beliefs": err_beliefs,
                "err_joint_beliefs": err_joint,
                "err_reward_beliefs": err_reward,
                "err_reward probabilities": err_reward_probabilities,
                "err_reward_beliefs_wrong": err_reward_beliefs_wrong,
                "err_surprise": err_surprise,
                "err_fisher": err_fisher,
                "err_deriv": err_deriv,
                "err_scrambled": err_reward_probabilities_scrambled,
            },
            "schedule candidates": supp,
            "dt": 0.5,
            "padding_time": 0.5,
        },
        f,
    )

# %%
ds = utils.data.open_pickle_file(os.path.join("../data", "analysis/bulk_beliefs.pkl"))
data, error, schedule_candidates, dt, pt = ds.values()
data |= {"ts_posterior": ts_posterior, "ts_reward_beliefs": ts_reward_belief}
error |= {
    "err_ts_posterior": err_ts_posterior,
    "err_ts_reward_belief": err_ts_reward_belief,
}
with open(os.path.join(SAVE_DIR, "bulk_beliefs.pkl"), "wb") as f:
    pickle.dump(
        {
            "data": data,
            "error": error,
            "schedule candidates": schedule_candidates,
            "dt": 0.5,
            "padding_time": 0.5,
        },
        f,
    )
