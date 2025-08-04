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
# This notebook provides a descriptive analysis of behavior under the abandoned exponential schedule and walks through observations that contextualize later analysis. As in the `Data Cleaning` notebook, the emphasis will be on the pushes instead of the continuous variables.

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from foraging.config.constants import KAPPA_LEVELS, PALETTE, PALETTE_DARK, SEED
from foraging.plotting import bp, enhanced_violinplot
from foraging.plotting.behavior import (
    plot_experiment_overview,
    plot_matching_law,
    plot_next_push_surprise,
    plot_push_intervals,
    plot_push_intervals_by_sessions,
    plot_push_intervals_vs_reward_intervals,
    plot_push_rates_across_block,
    plot_pushes,
    plot_quantity_across_block,
    plot_reward_per_push_across_block,
    plot_reward_rates_across_block,
    plot_runlengths,
    plot_stay_probabilities,
    plot_stay_switch_pushes,
)
from foraging.utils.data import (
    display_df,
    exclusion_criteria,
    filter_df,
    get_blocks,
    make_df,
)

# Filter out annoying matplotlib logs
mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)

# Constants
RNG = np.random.default_rng(SEED)
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.join(DATA_DIR, "experiments")
FIGURES_DIR = "../../figures"

# Modify the default kappa levels once
KAPPA_LEVELS["viktor"] = {"low": [0.01], "high": [0.1, 0.2]}

# %% [markdown]
# # Load Data
#
# The experiment data consists of multiple matfiles corresponding to the data for different subjects. Each matfile contains that subject's push, eye tracking (if available), and position (if available) data organized by blocks and sessions. Each block corresponds to a set of experiment parameters, notably the schedule of each box, the stimulus reliability kappa, and the stimulus type. A hierarchical overview of a given matfile is given in the `Data Cleaning` notebook.
#
# Here is a DataFrame showing a subset of the data, after applying the exclusion criteria detailed in `Data Cleaning`. Refer to the docstring of `make_df` and `exclusion_criteria` for more information about the contents of the DataFrame.

# %%
# %%capture --no-display
df = make_df(EXPERIMENT_DIR)
df = df.xs(1, level="shape", drop_level=False)
df = exclusion_criteria(df, EXPERIMENT_DIR)

# Remove some unnecessary blocks
df = df.drop(
    get_blocks(df[(df["schedule"] == 30) | (df["schedule"] == 7)]).size().index
)
df = df.drop(df[(df["schedule"] == 15) & (df["box"] == "medium")].index)
display_df(df, ["box", "push times", "reward outcomes"])

# %% [markdown]
# ## Data Overview
# A quick overview of the summary statistics:

# %%
print("schedules experienced by each subject")
print(df.groupby("subject")["schedule"].unique())

# %%
sns.histplot(df, x="duration")
plt.title("Duration of block")
plt.xlabel("duration (s)")

# %%
block_summary = (
    df.groupby(["subject", "session", "block"])
    .size()
    .reset_index(name="n pushes per block")
)
sns.violinplot(block_summary, x="subject", y="n pushes per block", cut=0)
plt.title("# pushes per block")

# %% [markdown]
# # Explore Data
# The behavioral quantities we will explore are: consecutive push intervals, wait times, stay pushes, switch pushes, runlengths of consecutive choices, etc. These are defined as follows:
# + <span style='color:#03a9fc'>consecutive push intervals</span>: time between consecutive pushes.
# + <span style='color:#fa5cbe'>wait times</span>: time between consecutive pushes *at the same box*-- can include pushes made at other boxes during this interval. This is essentially how long the subject has been waiting on this box to be ready before they push at that box again.
# + <span style='color:#fa0c9f'>stay times</span>: the intersection of consecutive push intervals and wait times, which occurs when the subject chooses to stay at their current box and push again.
# + <span style='color:#22c793'>switch times</span>: the time between consecutive pushes at different boxes, which will largely capture travel time between equidistant boxes.
# + <span style='color:#ff9d00'>runlengths</span>: the number of times the subject pushes the same box before switching boxes.
#
# One hypothesis we would like to test is whether the subjects' wait times obey the matching law, which is a normative principle that states that how often an option is picked is proportional to the reward they give. For example, a box with a faster mean schedule should be visited more frequently than a slower box. A good discussion of when matching is the result of theoretically optimal decision-making can be found in [(Sakai & Fukai 2008)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0003795). In our study, the matching law should materialize as the reward rates being proportional to the visit frequencies of each box; since each box has a different mean schedule, then, in expectation, the wait times for a given box should be proportional to the mean schedules of that box.
#
# ## Example Block

# %% [markdown]
# Here we show activity in an example block unfolding over time. Blue = fast box, yellow = medium box, red = slow box. Filled markers indicate rewarded pushes, unfilled unrewarded.

# %%
conds = dict(subject="viktor", session=20221011, block=3)
plot_pushes(df, conds, fig_kwargs=dict(figsize=(30, 2.2)), legend=False)

# %% [markdown]
# ## Pushes
#
# Here is an overview of the statistics of the pushes for each subject.

# %%
fig, ax = plt.subplots()
bp(sns.swarmplot)(
    df.groupby("subject").sample(10000, random_state=SEED, replace=True),
    x="subject",
    y="push intervals",
    hue="box",
    palette=PALETTE_DARK,
    legend=False,
    log_scale=True,
    size=0.25,
    dodge=True,
    ax=ax,
)
bp(enhanced_violinplot)(
    df,
    x="subject",
    y="push intervals",
    hue="box",
    palette=PALETTE,
    title_prefix="Distribution of push intervals at each box",
    y_unit="s",
    cut=0,
    inner=None,
    log_scale=True,
    common_norm=True,
    ax=ax,
)

# %% [markdown]
# Here are the distributions of wait times at each box as a function of stimulus reliability.

# %%
plot_push_intervals(df)

# %% [markdown]
# There is a striking bimodality in the wait time distributions that is preserved across subjects-- there is a fast mode and a slow mode.

# %%
plot_stay_switch_pushes(df)

# %% [markdown]
# First, for Dylan, the switch times and stay times are remarkably similar across all pairs of boxes. Generally speaking, for each subject the switch times are qualitatively similar, suggesting that most of the time spent pushing different boxes is travel time.  Second, for each subject, the stay times look qualitatively similar to each other as well.
#
# Finally, the distribution of consecutive push intervals can reveal the complexity of the decision process-- for example, suppose the decision space is {push at $\text{box}_i$ , wait}, and each timestep the animal makes a decision whether and where to push. Then, ignoring travel cost, a simple Hidden Markov Model (HMM) of this decision process would result in exponentially distributed intervals between pushes. Fitting an exponential distribution to each empirical distribution via maximum likelihood estimation and then performing a Kolmogorov-Smirnoff test under the null hypothesis that the data is drawn from the fitted exponential distributions resulted in rejecting the null hypothesis in each case, so the push intervals are likely the result of a much more complicated decision process.
#

# %% [markdown]
# ### Runlengths
# To get a sense of the statistics of *sequences* of pushes, we can gather the runlengths, that is the number of consecutive pushes at the same box before the subject switches.

# %%
plot_runlengths(df, stat="probability")

# %% [markdown]
# Aside from fitting in slightly more pushes at the fast box before switching when the reliability is high, there isn't much difference in runlengths between reliabilities.

# %% [markdown]
# ### Accuracy
#
# Here, we show each wait time along with the reward interval of the box that was pushed. Basically, we want to see how well the subjects timed their pushes to the reward interval of the box. Our expectation is that as the stimulus reliability increases, the subjects should be able to time their pushes more reliably to occur after the reward interval has elapsed.

# %%
plot_push_intervals_vs_reward_intervals(
    filter_df(df[(df["push intervals"] < 60) & (df["reward intervals"] < 60)]),
    annotate_reg=True,
)

# %% [markdown]
# Contrary to expectations, there appears to be no correlation between the wait times and reward intervals under any reliability conditions.

# %% [markdown]
# ### Surprise
#
# We can ask how much the factor of "surprise" at a particular reward outcome influences future behavior. For example, if the subject waits a long time and still doesn't get reward, they might 1) switch boxes and 2) wait longer to push the box they left if they start to believe it's a slower box.

# %%
# %%capture --no-display

plot_next_push_surprise(
    df,
    fig_kwargs=dict(figsize=(20, 10)),
    legend_kwargs=dict(loc="upper left", bbox_to_anchor=(1, 1), title=None),
)

# %% [markdown]
# Across both rewarded and unrewarded cases, there is a transition point at 20-25 s between waiting longer and going sooner the next push that is roughly preserved across all conditions. As reliability increases, the wait time on the x-axis clusters by boxes better and better. There is a distinct string of stay pushes that occur near the bottom of the graph where the subject pushes a few seconds after the current push. Visually, it appears that the subject switches a lot more across both reward outcomes, possibly switching more the longer they've waited, but when choosing to stay and push again, does so more after being rewarded. We confirm this below by counting how many times they stay or switch depending on how long they waited and the reward outcome.

# %% [markdown]
# ### Staying vs switching
#
# Here we calculate the probability of staying and pushing the same box again as a function of waiting time. We do this separately for the rewarded and unrewarded case.

# %%
plot_stay_probabilities(df, min_obs=10)

# %% [markdown]
# Clearly, the probability of staying is greater after receiving reward, but it generally goes down the longer the subject waits, with the exception of some weird quirks that could be due to low data volume. Tentatively, it even looks like the probability of staying grows more similar between reward outcomes as the subject waits longer.

# %% [markdown]
# ## Block Dynamics
#
# In this section, we explore how behavioral variables, such as reward rate and push frequency, evolve over time in the block. We will also compare to the optimal reward rate possible, if the subject were able to get each reward as soon as it became available. This is equal to the inverse mean schedules, and under independence of boxes and negligible travel costs, the sum of their individual reward rates yields the total reward rate.
#
# First, we'll examine the block initiation times to get a good sense of the right bin size for binning push times.

# %%
df_init = df.xs(1, level="push #")
bp(sns.violinplot)(
    df_init,
    x="subject",
    y="push times",
    cut=0,
    log_scale=True,
    title_prefix="Distribution of block initiation times",
    y_unit="s",
    legend=False,
)

# %% [markdown]
# ### Reward Rate

# %%
plot_reward_rates_across_block(df, min_obs=20)

# %% [markdown]
# The reward rates are very similar across subjects, starting off low and then increasing to a steady state that is only a bit lower than the gamma schedule! It is possible that even under this flawed paradigm, the subjects are still achieving some subjective reward rate.

# %%
# %%capture --no-display
plot_reward_rates_across_block(
    df,
    by_box=True,
    min_obs=20,
    show_traces=True,
)

# %% [markdown]
# The reward rates are somewhat ordered fast > medium > slow in Dylan and Marco's data and strongly ordered in Viktor's. There isn't an appreciable difference between reliabilities.
#
# ### Push Rate

# %%
plot_push_rates_across_block(df, by_box=True, min_obs=20, show_traces=True)


# %% [markdown]
# Dylan's push rates don't differentiate in the low reliability, but appears to do so somewhat in the high reliability. Marco also weakly differentiates the fast box from the slower boxes. Viktor differentiates the three boxes most strongly.

# %% [markdown]
# ### Push Intervals


# %%
def _auxiliary_plot(
    df: pd.DataFrame,
    conds: dict = None,
    palette: dict = None,
    ax: plt.Axes = None,
    **kwargs,
):
    schedules = (
        filter_df(df, {"stimulus reliability": "high", "subject": "viktor"})[
            ["box", "schedule"]
        ]
        .value_counts()
        .index.tolist()
    )
    for box, schedule in schedules:
        ax.axhline(schedule, color=palette[box], linestyle="--")
    ax.set_ylim(0, ax.get_ylim()[1] + 2)


plot_quantity_across_block(
    df,
    y="push intervals",
    auxiliary_plot=_auxiliary_plot,
    min_obs=10,
)

# %% [markdown]
# Dylan's and Marco's wait times are within range of the schedules but 1) fail to distinguish them when reliability is low 2) distinguish them weakly when the reliability is high. On the other hand, Viktor is pushing much more rapidly and distinguishing the schedules, for the same reward rate as the other two. This should translate to the other two monkeys achieving more accurate pushes than Viktor, which we confirm below.
#
# ### Reward-per-push

# %%
plot_reward_per_push_across_block(df, min_obs=10, by_box=True, average_blocks=True)

# %% [markdown]
# For Dylan and Marco, even though they pretty much push at the same push intervals across boxes, this naturally leads to an ordering in their accuracies going fast > medium > slow. They also have higher accuracies on average than Viktor, who has adopted a "spamming" strategy.

# %% [markdown]
# ## Matching Law
#
# The matching law is a behavioral principle that states that the number of times an option is picked should be proportional to how much reward is received at that option. For example, if option A gives twice as much reward as option B, then one should pick option A twice as much as they do option B. Formally:
#
# $$ \frac{C_i}{\sum_i C_i} \propto \frac{R_i}{\sum_i R_i}$$
#
# where $C_i$ and $R_i$ denote the number of times option $i$ has been chosen and reward obtained from that option, respectively. The constant of proportionality captures how sensitive the subjects are to the relative reward rates-- if the subject does not perceive much difference between the options, then the constant of proportionality will be less than 1 and the subject will be said to be "undermatching". When the constant of proportionality is greater than 1, then the subject is "overmatching". In the literature, undermatching is more common than overmatching, and also subjects have been observed to have a baseline response rate, or bias that cannot be attributed to reward. This bias and sensitivity to reward rates can be captured by linearly regressing the relative response rates on the relative reward rates, which is what we do below for each block with slopes and intercepts aggregated across blocks.

# %%
plot_matching_law(df)

# %% [markdown]
# There is a remarkable level of matching, even though some subjects adopt very simple strategies, such as waiting a fixed, long time to push. Matching can occur in these cases where the reward probability is close to 1 across boxes due to the long wait time, so then the relative reward rate is driven by the relative push rate.

# %% [markdown]
# # Conclusion
#
# To summarize:
# + Stimulus reliability has little effect on behavior
# + Reward rate does increase with time in block but is less than what is achievable under the Gamma schedule
# + Subjects weakly differentiate the boxes
# + Viktor seems to have found a strategy where he pushes faster than the other subjects but achieves the same reward rate overall, even at the cost of lowered accuracy per push
# + Matching can emerge even when subjects adopt simple "dumb" strategies such as waiting the same amount of time to push each box
