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
# This notebook provides a descriptive analysis of behavior under the exponential schedule and walks through observations that contextualize later analysis. As in the `Data Cleaning` notebook, the emphasis will be on the pushes instead of the continuous variables.

# %%
# %matplotlib inline

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from foraging.config.constants import (
    KAPPA_LEVELS,
    MULTIPLOT_FIGSIZE,
    PALETTE,
    PALETTE_DARK,
    SEED,
)
from foraging.plotting import (
    bp,
    enhanced_violinplot,
    plot_quantity_across_block,
    toggle_plot,
)
from foraging.plotting.behavior import (
    plot_experiment_overview,
    plot_matching_law,
    plot_next_push_surprise,
    plot_push_intervals,
    plot_push_intervals_by_sessions,
    plot_push_intervals_vs_reward_intervals,
    plot_push_rates_across_block,
    plot_pushes,
    plot_reward_per_push_across_block,
    plot_reward_rates_across_block,
    plot_runlengths,
    plot_stay_probabilities,
    plot_stay_switch_pushes,
)
from foraging.utils.autoreload import setup_auto_reload

# Create a new x-axis with regular intervals
from foraging.utils.data import (
    bin_data,
    display_df,
    exclusion_criteria,
    filter_df,
    get_blocks,
    make_df,
)

setup_auto_reload()

# Filter out annoying matplotlib logs
mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)

# Constants
RNG = np.random.default_rng(SEED)
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.join(DATA_DIR, "experiments")
FIGURES_DIR = "../../figures"
TO_HTML = True  # Change this to True when rendering HTML
ZOOM_KWARGS = {  # Convenience settings for zoomed in plots
    "show_traces": True,
    "smooth_kwargs": {"win_type": "gaussian", "window_size": 30, "step": 5},
}
FULL_BLOCK_KWARGS = {
    "min_obs": 10,
    "show_traces": True,
    "smooth_kwargs": {"win_type": "gaussian"},
}

# Modify the default kappa levels once
KAPPA_LEVELS["viktor"] = {"low": [0.01], "high": [0.1, 0.2]}

# %% [markdown]
# # Table of Contents
# - [Load Data](#Load-Data)
#   - [Data Overview](#Data-Overview)
# - [Explore Data](#Explore-Data)
#   - [Example Block](#Example-Block)
#   - [Pushes](#Pushes)
#     - [Runlengths](#Runlengths)
#     - [Accuracy](#Accuracy)
#     - [Surprise](#Surprise)
#     - [Staying vs Switching](#Staying-vs-Switching)
#   - [Block Dynamics](#Block-Dynamics)
#     - [Push Rate](#Push-Rate)
#     - [Push Intervals](#Push-Intervals)
#     - [Reward Rate](#Reward-Rate)
#     - [Reward-Per-Push](#Reward-Per-Push)
#   - [Matching Law](#Matching-Law)
# - [Conclusion](#Conclusion)

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

# Finally, chop off time from blocks that don't have much data
df = df[df["push times"] < 1200]
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
# The behavioral quantities we will explore are: push intervals, consecutive push intervals, stay pushes, switch pushes, runlengths of consecutive choices, etc. These are defined as follows:
# + <span style='color:#03a9fc'>push intervals</span>: time between pushes *at the same box*, which can include pushes made at other boxes during this interval. This is essentially how long the subject has been waiting on this box to be ready before they push at that box again. Depending on context, this is interchangeable with push times (for example, if discussing push times in a block, then this should be interpreted as the absolute time the push occurred in the block, not the time between pushes. On the other hand, push times for an individual choice the subject made should be interpreted as the push interval). Unless otherwise noted, this is the default meaning for push intervals.
# + <span style='color:#fa5cbe'>consecutive push intervals</span>: time between *consecutive* pushes-- unlike general push intervals, this can include the time between pushes at different boxes.
# + <span style='color:#fa0c9f'>stay times/stay pushes</span>: the intersection of consecutive push intervals and general push intervals, which occurs when the subject chooses to stay at their current box and push again.
# + <span style='color:#22c793'>switch times/switch pushes</span>: the time between consecutive pushes at different boxes, which may potentially largely capture travel time between equidistant boxes.
# + <span style='color:#ff9d00'>runlengths</span>: the number of times the subject pushes the same box before switching boxes.
#
# One hypothesis we would like to test is whether the subjects' push intervals obey the matching law, which is a normative principle that states that how often an option is picked is proportional to the reward they give. For example, a box with a faster schedule should be visited more frequently than a slower box. A good discussion of when matching is the result of theoretically optimal decision-making can be found in [(Sakai & Fukai 2008)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0003795). In our study, the matching law should materialize as the reward rates being proportional to the visit frequencies of each box; since each box has a different schedule, then, in expectation, the push times for a given box should be proportional to the schedules of that box.
#
# ## Example Block

# %% [markdown]
# Here we show activity in an example block unfolding over time. Blue = fast box, yellow = medium box, red = slow box. Filled markers indicate rewarded pushes, unfilled unrewarded.

# %%
conds = dict(subject="viktor", session=20221011, block=3)
plot_pushes(df, conds, fig_kwargs=dict(figsize=(30, 2.2)), legend=False)

# %% [markdown]
# Notice the sequences of pushes happening very close together in time. This is a salient feature of the behavior under the exponential schedule.

# %% [markdown]
# ## Pushes
#
# Here is an overview of the statistics of the pushes for each subject. Black dots connected by a line denote the mean push interval.

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
    title="Distribution of push intervals at each box",
    y_unit="s",
    cut=0,
    inner=None,
    log_scale=True,
    common_norm=True,
    ax=ax,
)

# %% [markdown]
# Here are those same distributions, now conditioned on stimulus reliability.

# %%
plot_push_intervals(
    df,
    cond_kwargs={
        "viktor": {
            "swarm_kwargs": {
                "size": 0.25,
            }
        }
    },
)

# %% [markdown]
# There is a striking bimodality in the wait time distributions that is preserved across subjects, consisting of a fast mode around ~1 sec and a slow mode around ~10 sec. Additionally, the mean push intervals differentiate the boxes and are ordered as expected (fast > medium > slow), but they do not seem to depend on stimulus reliability.
#
# Next, we visualize the distribution of stay and switch times.

# %%
plot_stay_switch_pushes(df)

# %% [markdown]
# First, for Dylan, the switch times and stay times are remarkably similar across all pairs of boxes. Generally speaking, for each subject the switch times are qualitatively similar, suggesting that most of the time spent pushing different boxes is travel time.  Second, for each subject, the stay times look qualitatively similar to each other as well.
#
# Finally, the distribution of consecutive push intervals can reveal the complexity of the decision process-- for example, suppose the decision space is {push at $\text{box}_i$ , wait}, and each timestep the animal makes a decision whether and where to push. Then, ignoring travel cost, a simple hidden Markov model of this decision process would result in exponentially distributed intervals between pushes. Fitting an exponential distribution to each empirical distribution via maximum likelihood estimation and then performing a Kolmogorov-Smirnoff test under the null hypothesis that the data is drawn from the fitted exponential distributions resulted in rejecting the null hypothesis in each case, so the push intervals are likely the result of a much more complicated decision process.
#

# %% [markdown]
# ### Runlengths
# To get a sense of the statistics of *sequences* of pushes, we take a look at the runlengths.

# %%
plot_runlengths(df, stat="probability")

# %% [markdown]
# Marco is the only subject whose runlengths show some sensitivity to stimulus reliability, fitting in slightly more pushes at the fast box when the reliability is high.

# %% [markdown]
# ### Accuracy
#
# Here, we show each push interval along with the reward interval of the box that was pushed. Basically, we want to see how well the subjects timed their pushes to the reward interval of the box. Our expectation is that as the stimulus reliability increases, the subjects should be able to time their pushes more reliably to occur after the reward interval has elapsed.

# %%
plot_push_intervals_vs_reward_intervals(
    filter_df(df[(df["push intervals"] < 60) & (df["reward intervals"] < 60)]),
    annotate_reg=True,
)

# %% [markdown]
# Contrary to expectations, there appears to be no correlation between the push intervals and reward intervals, not even when the reliability is high.

# %% [markdown]
# ### Surprise
#
# We can ask how much the factor of "surprise" at a particular reward outcome influences future behavior. For example, if the subject waits a long time and still doesn't get reward, they might 1) switch boxes and 2) wait longer to push the box they left if they start to believe it's a slower box. On the other hand, the behavior of an impatient subject who tends to spam pushes serially at the same box will manifest as a string of stay pushes straddling the y = -x line because y (change in push interval) + x (current push interval) = future push interval. Due to causality (future push can't come before current push), this line also marks the lower boundary of possible points.

# %%
plot_next_push_surprise(df, legend_kwargs=dict(title=None), s=10)

# %% [markdown]
# Rather idiosyncratic behavior. In general, the longer a subject waits to push, the sooner they will go next time, which is just a trivial aspect of the push interval distribution (to see this, consider a gaussian distribution and condition the differences between pairs of samples on one of the samples). Let's break this down:
#
# For Dylan, in almost all conditions, there is a string of spam pushes that are associated with *rapidly switching*, not staying. Only when the reliability is high and the push results in no reward does he opt to stay and push the same box, which is in contrast to other subjects' behavior where the string of spam pushes are almost exclusively stay pushes across all conditions. Another odd observation is that when Dylan decides to stay in all other conditions aside from (reliability = high, rewarded = no), he waits longer to push compared to the rapid switches. There are also a dense mode associated with short push intervals, where if he goes too fast and doesn't get reward, he appears to be extremely cautious and wait a long time next push. When reliability is low, he tends to stay and push again after a while, whereas when reliability is high he tends to switch before pushing.
#
# Marco's behavior is more typical, with a small cluster of points in the (reliability = low, rewarded = no) condition indicating a series of fast pushes.
#
# For Viktor, we see a few distinct clusters-- the string of fast stay pushes below all the points, cluster of switch pushes after fast pushes, and cloud of more reasonably paced switch pushes. These are present to various extents across all four conditions,

# %% [markdown]
# ### Staying vs Switching
#
# Here we calculate the probability of staying and pushing the same box again as a function of push interval aggregated across blocks. We do this separately for the rewarded and unrewarded case.

# %%
plot_stay_probabilities(df, min_obs=10)

# %% [markdown]
# Clearly, the probability of staying is greater after receiving reward, but it generally goes down the longer the subject waits, with the exception of some weird quirks that could be due to low data volume (especially for longer push intervals).

# %% [markdown]
# ## Block Dynamics
#
# In this section, we explore how behavioral variables, such as reward rate and push frequency, evolve over time in the block. We will also compare to the optimal reward rate possible, if the subject were able to get each reward as soon as it became available. This is equal to the inverse mean schedules, and assuming independent boxes and negligible travel costs, the sum of their individual reward rates yields the total reward rate.
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
    title="Distribution of block initiation times",
    y_unit="s",
    legend=False,
)

# %% [markdown]
# Looks like a bin size of 20-30 seconds should be big enough to prevent accidentally creating artifical transients in the first bin of each block. If we make the bin size too small, then we will be capturing trivial failures in the beginning of the block that aren't really reflective of anything important.
#
# In what follows, the rates are calculated by moving either a sliding rectangular or gaussian window over the relevant quantities in each block, with a window size of 60 seconds that slides every 5 seconds. The mean of each window is then divided by the size of the window to get a rate.

# %% [markdown]
# ### Push Rate

# %%
toggle_plot(
    plot_push_rates_across_block,
    plot_push_rates_across_block,
    button_labels=("Full block", "Zoomed in"),
    kwargs1={
        "df": df,
        "by_box": True,
        **FULL_BLOCK_KWARGS,
    },
    kwargs2={"df": df[df["push times"] < 120], "by_box": True, **ZOOM_KWARGS},
    inline=TO_HTML,
)


# %% [markdown]
# The push rates for Dylan and Marco are way lower than Viktor, who seems to be pushing twice as often as the other two. In addition, he differentiate the boxes better than the others; occasionally, we see a couple blocks where the other monkeys are differentiating the boxes, but these are the exception, not the norm.

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


def _plot(df, **kwargs):
    kwargs = kwargs.copy()
    smooth_kwargs = dict(
        fill_value=np.nan,
        min_periods=1,
    )

    kwargs.update({"smooth_kwargs": smooth_kwargs | kwargs.get("smooth_kwargs", {})})
    return plot_quantity_across_block(
        df[df["push intervals"] < 60],
        y="push intervals",
        x="push times",
        auxiliary_plot=_auxiliary_plot,
        by_box=True,
        fig_title="Push intervals",
        **kwargs,
    )


toggle_plot(
    _plot,
    _plot,
    button_labels=("Full block", "Zoomed in"),
    kwargs1={
        "df": df,
        **FULL_BLOCK_KWARGS,
    },
    kwargs2={"df": df[df["push times"] < 120], **ZOOM_KWARGS},
    inline=TO_HTML,
)

# %% [markdown]
# Dylan's and Marco's push intervals are within range of the schedules, but Viktor is pushing much more rapidly while distinguishing the schedules better, for an even better reward rate. This highlights one of the quirks of the exponential schedule, namely that reward rate can be somewhat maximized by a less-than-ideal strategy. We will verify in later theoretical analysis what the optimal reward rate looks like, and how the subjects compare.

# %% [markdown]
# ### Reward Rate

# %%
toggle_plot(
    plot_reward_rates_across_block,
    plot_reward_rates_across_block,
    button_labels=("Full block", "Zoomed in"),
    kwargs1={"df": df, **FULL_BLOCK_KWARGS},
    kwargs2={"df": df[df["push times"] < 120], **ZOOM_KWARGS},
    inline=TO_HTML,
)

# %% [markdown]
# Viktor also achieves double the reward rate that the other subjects achieve. Note that this is still lower than what he achieves under the gamma schedule.

# %%
toggle_plot(
    plot_reward_rates_across_block,
    plot_reward_rates_across_block,
    button_labels=("Full block", "Zoomed in"),
    kwargs1={
        "df": df,
        "by_box": True,
        **FULL_BLOCK_KWARGS,
    },
    kwargs2={"df": df[df["push times"] < 120], "by_box": True, **ZOOM_KWARGS},
    inline=TO_HTML,
)

# %% [markdown]
# The reward rates are somewhat ordered fast > medium > slow in Dylan and Marco's data and more strongly ordered in Viktor's.
# Next, we will show the reward-per-push, which can be interpreted as the probability of getting reward. We expect Viktor to do poorly under this metric.
#
# ### Reward-Per-Push

# %%
toggle_plot(
    plot_reward_per_push_across_block,
    plot_reward_per_push_across_block,
    button_labels=("Full block", "Zoomed in"),
    kwargs1={
        "df": df,
        "by_box": True,
        **FULL_BLOCK_KWARGS,
    },
    kwargs2={"df": df[df["push times"] < 120], "by_box": True, **ZOOM_KWARGS},
    inline=TO_HTML,
)

# %% [markdown]
# While there is a lot of variability, it is clear that on average Viktor is less accurate than the others.

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
# There is a remarkable level of matching, even when subjects don't really differentiate the boxes. Matching can occur in these cases when the reward probability is close to 1 across boxes due to waiting a long time to push, so then the relative reward rate is driven by the relative push rate.

# %% [markdown]
# # Conclusion
#
# To summarize:
# + Stimulus reliability has little effect on behavior
# + Subjects, at best, weakly differentiate the boxes
# + Viktor seems to have found a strategy where he pushes faster than the other subjects, achieving low accuracy but much higher reward rate
# + Matching can emerge even when subjects don't differentiate the boxes
