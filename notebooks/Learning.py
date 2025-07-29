# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
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
from scipy.special import polygamma

from foraging import plotting, utils
from foraging.config.constants import BOX_COLORS, BOX_LABELS, PALETTE, SEED
from foraging.plotting import bp, format_yticks, per_block, titler
from foraging.plotting.beliefs import (
    plot_cramer_rao_lb,
    plot_fisher_info,
    plot_schedule_beliefs_in_block,
    plot_schedule_beliefs_mean_and_std_across_blocks,
)
from foraging.utils import INDEX, MIN_INDEX
from foraging.utils.beliefs import (
    compute_posteriors,
    get_mean_beliefs_over_time,
    get_std_beliefs_over_time,
)
from foraging.utils.data import (
    display_df,
    exclusion_criteria,
    filter_df,
    make_df,
    process_blocks,
)
from foraging.utils.models import (
    GammaBoxBelief,
    GammaBoxPermutationBelief,
    IndependentBoxesBelief,
)

pd.options.mode.copy_on_write = True

# constants
RNG = np.random.default_rng(SEED)
DATA_DIR = "../data"
EXPERIMENT_DIR = os.path.join(DATA_DIR, "experiments")

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
# # Theoretical Beliefs about the Schedules
#
# In theory, observations about reward carries information about the task parameters that generate them. For example, if the subject is able to perfectly observe the reward intervals $\{t_{\text{reward}}\}$, which is the time that needs to elapse before reward becomes available, then they can average these reward intervals to get an estimate of the box's schedule $\lambda$, the mean of the generative distribution. It is known that the sample mean of observations drawn from a Gaussian distribution is the most efficient estimator of the population mean, in the sense that the variance achieves the Cramer-Rao lower bound. The same is true for the sample mean of observations drawn from a gamma distribution, which is the distribution we use in the experiment to generate reward intervals.
#
# Now, in the actual experiment, the rewards are hidden when they become available and the subject only receives potentially noisy observations of the reward interval. The observations take the form of a color cue that encodes the reward interval with some spatiotemporal noise. On one end of the spectrum lie completely unreliable observations that offer no information about the reward intervals, and hence no information about the schedules that generated them; however, there is still hope in the form of censored observations made available when the subject decides to push at the box. Then, the subject observes whether reward was available by the time they pushed at $t_{\text{push}}$, and they can use this observation to update their beliefs about the schedules. For example, if the subject waits a long time at a box they believe to be fast but still does not observe reward when they push, that is an indication that the box may be slower than anticipated, and vice versa if the subject only waits a short time at a box they thought was slow and is pleasantly surprised to receive reward. Both the perfect observations and the censored observations correspond to distinct but related likelihood functions, described below:
#
# *Perfect Observations*
#
# $$
# L(\lambda) = f(t_{\text{reward}}|\lambda, \alpha)
# $$
#
# *Censored Observations*
#
# $$
# L(\lambda) = \begin{cases}
# F(t_{\text{push}} | \lambda, \alpha) & \text{if reward }\\
# 1 - F(t_{\text{push}} | \lambda, \alpha) & \text{else}
# \end{cases}
# $$
#
# where $f$ and $F$ are the pdf and cdf of the gamma distribution, respectively, and $\alpha$ denotes the shape parameter, which we assume to be known $^1$. From these, we can derive lower bounds on how fast the uncertainty can shrink for the ideal observer. Assuming a continuous hypothesis space over possible schedules, and independence between boxes, one way to do this is to derive the Fisher information, which we go into next section.
#
# ## Fisher Information
#
# The Fisher information tells us how discriminable the task parameters, such as the schedules $\{\lambda_i\}$, of the experiment are from similar parameters-- the higher the Fisher information, the higher the theoretical limit on our knowledge or certainty about the parameters. This relationship is formally known as the Cramer-Rao lower bound, which states that the variance of any unbiased estimator can be no lower than the inverse Fisher information. Assuming the shape parameter $\alpha$ is known, the Fisher information of perfect and censored obserations is given below:
#
# *Perfect Observations*
#
# $$J(\lambda) = \frac{\alpha}{\mu^2}$$
#
# *Censored Observations*
#
# $$J(\lambda) = (\frac{t_{\text{push}}}{\lambda})^2 \frac{f^2(t_{\text{push}}|\lambda, \alpha)}{F(t_{\text{push}}|\lambda, \alpha)(1-F(t_{\text{push}}|\lambda, \alpha))}$$
#
# Notice that the Fisher information of the censored observations depends on the push time $t_{\text{push}}$. Intuitively, this is because if you wait too long to push relative to the schedule or go too soon, then on average you will get very little information. Only a moderate regime of wait times gives useful information, and each box has a different optimal wait time that delivers maximum Fisher information. These curves are shown below.
#
# *The shape parameter basically influences the variance of the reward intervals and is held constant across blocks as well as shared between boxes. At the end of this notebook, we derive the Cramer-Rao lower bound for the shape parameter and show that it virtually vanishes to 0 after accumulating a typical block's worth of observations, lending credence to our assumption that the subjects know the shape parameter to a high degree of confidence.

# %%
plot_fisher_info(
    np.linspace(1, 30, 100),
    schedules=[7, 14, 21],
    alpha=10,
    title="Fisher information of Reward Observations",
)

# %% [markdown]
# The optimal wait times that maximize the Fisher information of each box occur at distinct timepoints ordered by schedules. Notice that the maximal Fisher information of the two slower boxes are close together in value compared to the fast box, suggesting that the fast box may be easiest to figure out first because it only needs faster and fewer pushes to reach a certain precision. Using the maximal Fisher information of each box, we show how the uncertainty shrinks with more observations by showing how the standard deviation, the square root of the Cramer-Rao lower bound, decreases with the number of obervations.

# %%
plot_cramer_rao_lb(n=50, schedules=[7, 14, 21], alpha=10)


# %% [markdown]
# After 50 pushes, all schedules can be estimated within ±1-2 s. Note that these results are shown for shape parameter = 10; when shape = 1, it takes 3-4x longer to achieve the same precision.

# %% [markdown]
# ## What if the hypothesis space is discrete and finite?
#
# Now, this framework is only applicable if we assume the subjects 1) treat the boxes as independent and 2) start their inference process from scratch at the start of each block. If the schedules are kept constant over several blocks and merely permuted, then it is conceivable that the subjects eventually get a good handle on the identities of the schedules and only infer the assighment of specific schedules to boxes, not the values of the schedules themselves. In that case, their beliefs are defined over $m!$ permutations, where $m$ is the number of boxes. Analogous to the Fisher information, one can simulate observations under one of the permutations and show how quickly the posterior converges to the right permutation as a function of the number of observations (the result will generalize to the other permutations).

# %% [markdown]
# ## Empirical Beliefs using Censored Observations
#
# Given real behavioral data, we can construct Bayes-optimal beliefs about the schedules using censored observations when the color cue is completely unreliable. These beliefs take the form of a posterior over possible schedules, and they are updated according to Bayes rule:
#
# $$b(\lambda|o_n := t_{\text{push}}) \propto \begin{cases} F(t_{\text{push}} | \lambda)b(\lambda|o_{1:n-1}) & \text{if reward}\\ (1-F(t_{\text{push}} | \lambda))b(\lambda|o_{1:n-1}) &  \text{else}\end{cases}$$
#
# Assuming a uniform prior over possible schedules that resets between blocks, the maximum a posteriori (MAP) estimator is equivalent to the maximum likelihood estimator (MLE). An example posterior over one block is shown below.


# %%
# Create a posterior class that is independent for each box
class Posterior(IndependentBoxesBelief):
    def __init__(self, n_boxes: int, *args, **kwargs):
        super().__init__(n_boxes, GammaBoxBelief, *args, **kwargs)


schedule_candidates = np.arange(30) + 1
schedule_beliefs_shape10, _ = process_blocks(
    filter_df(df, {"kappa": 0, "shape": 10}),
    compute_posteriors,
    Posterior,
    schedules=schedule_candidates,
    shape=10,
)
schedule_beliefs_shape1, _ = process_blocks(
    filter_df(df, {"kappa": 0, "shape": 1}),
    compute_posteriors,
    Posterior,
    schedules=schedule_candidates,
    shape=1,
)
schedule_beliefs = schedule_beliefs_shape10 | schedule_beliefs_shape1
conds = dict(subject="viktor", session=20230807, block=5)
plot_schedule_beliefs_in_block(df, schedule_beliefs, conds)

# %%
schedule_beliefs_shape10, _ = process_blocks(
    filter_df(df, {"kappa": 0, "shape": 10}),
    compute_posteriors,
    GammaBoxPermutationBelief,
    schedules=[7, 14, 21],
    shape=10,
)

# %% [markdown]
# Now we will aggregate posteriors over blocks and report the average summary statistics of those posteriors.

# %%
plot_schedule_beliefs_mean_and_std_across_blocks(
    df,
    schedule_beliefs_shape10,
    conds={"kappa": 0, "shape": 10},
    x="push # by box",
    min_obs=5,
)

# %% [markdown]
# # Extra: Knowing the shape parameter after a couple blocks of behavior
#
# So far, we have focused on infering the schedule independently of the shape parameter $\alpha$ that also parameterizes the reward interval distribution. Given that this parameter is fixed across boxes and blocks, one might expect that the subjects are capable of inferring the shape parameter with high confidence only after a couple blocks. To materialize this expectation, we can derive the Fisher information to get a theoretical lower bound on the uncertainty as a function of the number of observations the subject makes.
#
# Under the standard parameterization of the Gamma distribution, which uses the scale parameter $\theta:=\frac{\mu}{\alpha}$, the Fisher information matrix for a single perfect observation of reward interval $t_{\text{reward}}$ is:
#
# $$J(\alpha, \theta) =
# \begin{bmatrix}
# \psi^{(1)}(\alpha) & \frac{1}{\theta}\\
# \frac{1}{\theta} & \frac{\alpha}{\theta^2}
# \end{bmatrix}
# $$
#
# where $\psi^{(1)}$ denotes the trigamma function. We'll omit the censored obserations for now since they will give similar behavior. We visualize how the Cramer-Rao lower bound shrinks with the number of independent observations below:


# %%
def fisher_info_shape(mu, alpha):
    scale = mu / alpha
    mat = np.array([[polygamma(1, alpha), 1 / scale], [1 / scale, alpha / scale**2]])
    return mat[-1, -1]


# Cramer-Rao lower bound for standard deviation as a function of number of observations
means = [7, 14, 21]
shape = 10
n_obs = np.arange(200) + 1
colors = list(PALETTE.values())
labels = list(PALETTE.keys())
for i, mu in enumerate(means):
    fisher_info = fisher_info_shape(mu, shape)
    std_crlb = np.sqrt(1 / (n_obs * fisher_info))
    plt.plot(n_obs, std_crlb, label=labels[i], color=colors[i])

plt.xlabel("# observations")
plt.ylabel("standard deviation")
plt.title("Uncertainty about shape parameter as a function of # observations")
plt.legend()
plt.show()

# %% [markdown]
# A typical block of pushes may contain around 200 pushes, so we can safely say that the ideal observer can figure out the shape parameter to a precision of ± 0.1 after just one block of perfect observations about the reward intervals. Also note a curve is shown for each box, but one can imagine pooling observations across all boxes and achieving even lower bounds on the uncertainty.

# %% [markdown]
# ## Bias
#
# It's perfectly possible that the subjects have a prior about the schedules that could be mismatched to the true schedules. Now, asymptotically the posterior converges to a Gaussian centered on the MLE, with variance equal to 1/fisher information (Laplace approximation), but in the beginning a nontrivial bias may be present that influences behavior. It is also worth thinking about constrained computation-- if the agent prematurely stops updating their posterior due to computational budget, then they may also be biased in their belief about the schedule depending on the nature of their limited observations. Depending on the nature of the bias, the MSE may even be less than that of the unbiased estimator (bias-variance decomposition)
