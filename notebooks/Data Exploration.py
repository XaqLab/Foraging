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
# This notebook provides a descriptive analysis of behavior and walks through observations that contextualize later analysis. As in the `Data Cleaning` notebook, the emphasis will be on the pushes instead of the continuous variables, like gaze and position.

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

from foraging.config.constants import MULTIPLOT_FIGSIZE, PALETTE, PALETTE_DARK, SEED
from foraging.plotting import bp, enhanced_violinplot
from foraging.plotting.behavior import (
    plot_accuracy_across_block,
    plot_experiment_overview,
    plot_matching_law,
    plot_matching_law_coefficients,
    plot_next_push_surprise,
    plot_push_intervals,
    plot_push_intervals_by_sessions,
    plot_push_intervals_vs_reward_intervals,
    plot_push_rates_across_block,
    plot_pushes,
    plot_quantity_across_block,
    plot_reward_rates_across_block,
    plot_runlengths,
    plot_stay_probabilities,
    plot_stay_switch_pushes,
)
from foraging.utils.data import display_df, exclusion_criteria, filter_df, make_df

# Filter out annoying matplotlib logs
mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)

# Constants
RNG = np.random.default_rng(SEED)
DATA_DIR = "../data"
EXPERIMENT_DIR = os.path.join(DATA_DIR, "experiments")
FIGURES_DIR = "../figures"

# %% [markdown]
# # Load Data
#
# The experiment data consists of multiple matfiles corresponding to the data for different subjects. Each matfile contains that subject's push, eye tracking (if available), and position (if available) data organized by blocks and sessions. Each block corresponds to a set of experiment parameters, notably the schedule of each box, the stimulus reliability kappa, and the stimulus type. A hierarchical overview of a given matfile is given in the `Data Cleaning` notebook.
#
# Here is a DataFrame showing a subset of the data, after applying the exclusion criteria detailed in `Data Cleaning`. Refer to the docstring of `make_df` and `exclusion_criteria` for more information about the contents of the DataFrame.

# %%
# %%capture --no-display
df = make_df(os.path.join(DATA_DIR, "experiments"))
df = exclusion_criteria(df, EXPERIMENT_DIR)
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
# Here is a bird's eye view of one subject's behavioral data over the course of the entire experiment. Each row is a session of consecutive blocks. Each block's pushes are displayed as a raster plot over the duration of that block, and the pushes at each box position are stacked on top of each other (top to bottom is 3, 2, 1). Finally, pushes are colored by box schedule, so we can see how the schedules are assigned to each box over consecutive blocks.

# %%
conds = dict(subject="viktor")
plot_experiment_overview(
    df,
    conds=conds,
    title_prefix="Overview of pushes over entire experiment",
    annotate_block=True,
)

# %% [markdown]
# # Explore Data
# The behavioral quantities we will explore are: push intervals, consecutive push intervals, stay pushes, switch pushes, runlengths of consecutive choices, etc. These are defined as follows:
# + <span style='color:#03a9fc'>push intervals</span>: time between pushes *at the same box*, which can include pushes made at other boxes during this interval. This is essentially how long the subject has been waiting on this box to be ready before they push at that box again. Depending on context, this is interchangeable with push times (for example, if discussing push times in a block, then this should be interpreted as the absolute time the push occurred in the block, not the time between pushes. On the other hand, push times for an individual choice the subject made should be interpreted as the push interval). Unless otherwise noted, this is the default meaning for push intervals.
# + <span style='color:#fa5cbe'>consecutive push intervals</span>: time between *consecutive* pushes-- unlike general push intervals, this can include the time between pushes at different boxes.
# + <span style='color:#fa0c9f'>stay times</span>: the intersection of consecutive push intervals and general push intervals, which occurs when the subject chooses to stay at their current box and push again.
# + <span style='color:#22c793'>switch times</span>: the time between consecutive pushes at different boxes, which may potentially largely capture travel time between equidistant boxes.
# + <span style='color:#ff9d00'>runlengths</span>: the number of times the subject pushes the same box before switching boxes.
#
# One hypothesis we would like to test is whether the subjects' push intervals obey the matching law, which is a normative principle that states that how often an option is picked is proportional to the reward they give. For example, a box with a faster schedule should be visited more frequently than a slower box. A good discussion of when matching is the result of theoretically optimal decision-making can be found in [(Sakai & Fukai 2008)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0003795). In our study, the matching law should materialize as the reward rates being proportional to the visit frequencies of each box; since each box has a different schedule, then, in expectation, the push times for a given box should be proportional to the schedules of that box.
#
# ## Example Block

# %% [markdown]
# Here we show activity in an example block unfolding over time. Blue = fast box, yellow = medium box, red = slow box. Filled markers indicate rewarded pushes, unfilled unrewarded.

# %%
conds = dict(subject="viktor", session=20230811, block=3)
plot_pushes(df, conds, fig_kwargs=dict(figsize=(30, 2.2)), legend=False)

# %% [markdown]
# ## Pushes
#
# Here is an overview of the statistics of the pushes for each subject.

# %%
fig, ax = plt.subplots(figsize=MULTIPLOT_FIGSIZE)
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
# There is a bimodality present in a couple subjects. Below, we split the data into the exponential schedule and gamma schedule.

# %%
fig, ax = plt.subplots(figsize=MULTIPLOT_FIGSIZE)
bp(sns.swarmplot)(
    df.xs(1, level="shape")
    .groupby("subject")
    .sample(10000, random_state=SEED, replace=True),
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
    df.xs(1, level="shape"),
    x="subject",
    y="push intervals",
    hue="box",
    palette=PALETTE,
    title_prefix="Distribution of push intervals under exponential schedule",
    y_unit="s",
    cut=0,
    inner=None,
    log_scale=True,
    common_norm=True,
    ax=ax,
)

# %%
fig, ax = plt.subplots(figsize=MULTIPLOT_FIGSIZE)
bp(sns.swarmplot)(
    df.xs(10, level="shape")
    .groupby("subject")
    .sample(10000, random_state=SEED, replace=True),
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
    df.xs(10, level="shape"),
    x="subject",
    y="push intervals",
    hue="box",
    palette=PALETTE,
    title_prefix="Distribution of push intervals under gamma schedule",
    y_unit="s",
    cut=0,
    inner=None,
    log_scale=True,
    common_norm=True,
    ax=ax,
)

# %% [markdown]
# It's clear the bimodality results from some property of the exponential schedule. To confirm this and see if there are any other trends that emerge throughout the experiment, we'll next show the distribution of push intervals in each session. Since humans do not have session data, they are omitted for now.

# %%
# %%capture --no-display
plot_push_intervals_by_sessions(df)

# %% [markdown]
# These swarmplots are useful for visualizing individual data points by jittering the ones that collide, to the limit that the data points aren't too cluttered and impede comprehension. The x-axis denotes days since the first session (with day 0 corresponding to the first session) and the y-axis are the log push intervals in each session. For Dylan and Marco, the number of data points is manageable but for Viktor they are so numerous that you can see where the plot struggles to fit all the data points. Each swarmplot is preceded by an experiment overview where, for each session, we denote how many blocks had a certain parameter value, which gives us a sense of the timeline for how the experiment evolved and when decisions were made to change the experiment. The x-axes for a swarmplot and its corresponding experiment overview are aligned for ease of comparison.
#
# One thing to look for are sessions that look distributionally different from the others, e.g. day 0 for Dylan shows a bump around extremely fast push intervals around 1 second, and days 53-59 contain a big bump of pushes right above 1 second. For Marco, day 4 is extremely scarce in pushes compared to other days. It's interesting to see heterogeneous patterns in the pushes, in particular for Viktor. In fact, we see three distinct patterns of clusters organized in time: days 0-91 exhibit strong bimodality, days 100-129 are more unimodal and concentrate towards larger push intervals, and days 300-317 concentrate on even larger push intervals. Notice that day 300 coincides with a change in the experiment from the exponential schedule to gamma schedule.
#
# Since the exponential schedule depicts atypical behavior, this notebook will be dedicated to the gamma schedule and a supplementary notebook will focus on the exponential schedule.

# %%
df = filter_df(df, {"shape": 10})

# %% [markdown]
# Here are the distributions of push intervals at each box as a function of stimulus reliability.

# %%
plot_push_intervals(df)

# %% [markdown]
# The distributions are very similar across stimulus reliabilities, suggesting subjects are able to differentiate the boxes to a similar degree regardless of task difficulty. The distributions seem to grow slightly sharper as reliability increases.
#
# Next, we show the distribution of consecutive push intervals between all pairs of boxes simultaneously, similar to a transition matrix where the rows are the boxes the subject first pushes and the columns are where they push next.

# %%
plot_stay_switch_pushes(df)

# %% [markdown]
# First, the switch times are qualitatively similar across all pairs of boxes. Second, the stay times across all boxes share bimodal features covering about the same range, but the fast box looks qualitatively different from the other boxes.
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
# Here, we show each push interval along with the corresponding reward interval of the box that was pushed. Basically, we want to see how accurately the subjects timed their pushes to the reward interval of the box. Our expectation is that as the stimulus reliability increases, the subjects should be able to time their pushes more reliably to occur after the reward interval has elapsed.

# %%
plot_push_intervals_vs_reward_intervals(
    filter_df(df[(df["push intervals"] < 60) & (df["reward intervals"] < 60)]),
    annotate_reg=True,
)

# %% [markdown]
# As expected, the correlation and slope between the push intervals and reward intervals increases with stimulus reliability.

# %% [markdown]
# #TODO: behavioral adaptation between blocks with different stimulus reliability? for example, one might expect that after adapting to an unreliable block, there is a reliance on internal model over color for some time before animal adjusts to new block statistics

# %% [markdown]
# ### Surprise
#
# We can ask how much the factor of "surprise" at a particular reward outcome influences future behavior. For example, if the subject waits a long time and still doesn't get reward, they might 1) switch boxes and 2) wait longer to push the box they left if they start to believe it's a slower box.

# %%
# %%capture --no-display
plot_next_push_surprise(
    df,
    legend_kwargs=dict(title=None),
)

# %% [markdown]
# Across both rewarded and unrewarded cases, there is a transition point at 20-25 s between waiting longer and going sooner the next push that is roughly preserved across all conditions. As reliability increases, the push intervals on the x-axis cluster by boxes. There is a distinct string of stay pushes that occur near the bottom of the graph where the subject pushes a few seconds after the current push. Visually, it appears that the subject switches a lot more across both reward outcomes, possibly switching more the longer they've waited, but when choosing to stay and push again, does so more after being rewarded. We confirm this below by counting how many times they stay or switch depending on how long they waited to push and the reward outcome.

# %% [markdown]
# ### Probability of staying
#
# Here we calculate the probability of staying and pushing the same box again as a function of push interval. We do this separately for the rewarded and unrewarded case.

# %%
plot_stay_probabilities(df, min_obs=10)

# %% [markdown]
# Clearly, the probability of staying is greater after receiving reward, but it generally goes down the longer the subject waits, with the exception of some weird quirks that could be due to low data volume. Tentatively, it even looks like the probability of staying grows more similar between reward outcomes as the subject waits longer.

# %% [markdown]
# ## Block Dynamics
#
# In this section, we explore how behavioral variables, such as reward rate and push frequency, evolve over time in the block. We will also compare to the optimal reward rate possible, if the subject were able to get each reward as soon as it became available. This is equal to the inverse of the schedules, and under negligible travel costs, the sum of their individual reward rates yields the maximum total reward rate.
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
# Looks like a bin size of 20-30 seconds should be big enough to prevent accidentally creating artifical transients.
#
# ### Reward Rate

# %%
plot_reward_rates_across_block(df, min_obs=10)

# %% [markdown]
# Reward rate increases as subjects gain more experience in the block. As reliablity increases, the total reward rate combined across all boxes also increases. Next, we decompose the reward rate into different boxes.

# %%
plot_reward_rates_across_block(
    df,
    by_box=True,
    min_obs=10,
)

# %% [markdown]
# Clearly, the reward rates are ordered the way we would expect them, fast > medium > slow.

# %% [markdown]
# ### Push Rate

# %%
plot_push_rates_across_block(
    df,
    by_box=True,
    min_obs=10,
)


# %% [markdown]
# When the stimulus reliability is low, the subjects initially struggle to distinguish the boxes. As time goes on, they learn.

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


plot_quantity_across_block(
    df,
    y="push intervals",
    auxiliary_plot=_auxiliary_plot,
    min_obs=10,
)

# %% [markdown]
# Across all reliabilities, the push intervals start off low and close together before differentiating as the block progresses. When reliability is low, the push intervals are similar across boxes and differentiation occurs slowly compared to higher reliabilities. When reliability is high, the subjects's pushes are guided by the color cue, and so their pushes distinguish the boxes quickly. Next, we will see if how this translates to accuracy by counting the fraction of rewarded pushes in each time bin across blocks.
#
# ### Accuracy

# %%
plot_accuracy_across_block(df, min_obs=10)

# %% [markdown]
# The accuracies start off low but increase over time in the block. As reliability increases, they start off higher.

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

# %%
# %%capture --no-display
plot_matching_law(df, palette="rocket", min_obs=10)

# %% [markdown]
# Across all subjects, there is undermatching as indicated by slopes < 1 and bias that improves with reliability.
#

# %% [markdown]
# # Conclusion
#
# To summarize:
# + Subjects become more accurate as reliability increases
# + Initially in the block, reward rates across boxes are low but increase and differentiate with experience in the block
# + Undermatching and bias improve with reliability
