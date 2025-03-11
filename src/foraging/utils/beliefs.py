import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.typing import ArrayLike
from scipy.stats import uniform

import models
from ._base import discrete_time
from .data import process_block_safely

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def compute_posteriors(
        df: pd.DataFrame,
        index: tuple,
        schedule_candidates: ArrayLike,
        record: bool = True
) -> models.IndependentBoxesPosterior:
    """
    Computes the posterior belief over reward schedules for each box, updating after each push.

    Args:
        df (pd.DataFrame): Dataframe containing session data.
        index (tuple): Multi-index selector for retrieving block-specific data.
        schedule_candidates (ArrayLike): Array of possible reward rates forming the support of the beliefs.
        record (bool, optional): Whether to store posterior history. Defaults to True.

    Returns:
        models.IndependentBoxesPosterior: The computed posterior belief over reward schedules.
    """

    # Extract data corresponding to the given index
    block_data = df.loc[index]

    # Get unique schedules (sorted in descending order)
    schedules = np.sort(block_data['schedule'].unique())[::-1]
    n_boxes = len(schedules)

    # Assume agent knows the exact number of states
    shape = block_data.index.unique('shape')[0]

    # Construct prior using a uniform distribution over schedule candidates
    prior = uniform.pdf(
        schedule_candidates,
        loc=schedule_candidates.min(),
        scale=schedule_candidates.max() - schedule_candidates.min()
    )

    # Normalize the prior
    normalizer = models.MyNormalizer()
    prior = normalizer.normalize(prior)

    # Initialize observation model
    obs_model = models.GammaObservation(shape)

    # Construct posterior belief model
    posterior = models.IndependentBoxesPosterior(
        n_boxes, obs_model, prior, schedule_candidates, normalizer, record=record
    )

    # Iterate over each box in order from slowest to fastest
    for i in range(n_boxes):
        box_mask = block_data['box rank'] == i

        push_times = block_data.loc[box_mask, 'push times'].values
        push_intervals = block_data.loc[box_mask, 'same-box push intervals'].values
        reward_outcomes = block_data.loc[box_mask, 'reward outcomes'].values

        # Number of non-NaN observations
        n_obs = np.count_nonzero(~np.isnan(push_times))

        # Update posterior using each valid observation
        for t in range(n_obs):
            if not np.isnan(push_intervals[t]):
                posterior.update((reward_outcomes[t], push_intervals[t]), i)

    return posterior

@process_block_safely
def compute_latent_beliefs_over_time(
        df: pd.DataFrame,
        index: tuple,
        posterior: models.Posterior,
        dt: float = 0.5,
        padding_time: float = 0.5
) -> np.ndarray[float]:
    """
    Compute the latent belief over time regarding the reward schedule for each box.

    Args:
        df: Pandas DataFrame containing session data.
        index: Index to locate the relevant block data.
        posterior: Instance of the Posterior class containing belief updates.
        dt: Time bin size for discretizing the continuous-time beliefs.
        padding_time: Additional time after the last push for modeling beliefs.

    Returns:
        np.ndarray: A time-based belief array of shape (n_boxes, n_timesteps, len(posterior.support())).
    """

    block_data = df.loc[index]
    end_t = block_data['push times'].max() + padding_time
    n_boxes = posterior.n_boxes
    n_timesteps = int(end_t / dt)

    schedule_belief_time = np.zeros((n_boxes, n_timesteps, len(posterior.support())))

    for i in range(n_boxes):
        box_idx = block_data['box rank'] == i
        push_times = block_data.loc[box_idx, 'push times'].to_numpy()
        push_times = push_times[~np.isnan(push_times)]

        # Convert posterior to discrete-time representation
        schedule_belief_time[i] = discrete_time(posterior.boxes[i].history, push_times, dt, end_t)

    return schedule_belief_time

@process_block_safely
def compute_reward_beliefs(
        df: pd.DataFrame,
        index: tuple,
        posterior: models.Posterior
) -> np.ndarray:
    """
    Compute the belief of reward availability for each box as an event-based representation.
    This extracts the belief about reward availability immediately before each push.

    Args:
        df: Pandas DataFrame containing session data.
        index: Index to locate the relevant block data.
        posterior: Instance of the Posterior class containing belief updates.

    Returns:
        np.ndarray: An event-based belief array of shape (n_obs, n_boxes, 2),
                    where beliefs are evaluated before each push.
    """

    n_boxes = posterior.n_boxes
    block_data = df.loc[index]

    # Compute availability marginal for each push
    n_obs = block_data['push times'].size - np.count_nonzero(np.isnan(block_data['push times']))
    belief_avail_event = np.zeros((n_obs, n_boxes, 2))
    push_times_and_box = block_data[['push times', 'box rank']].values[:n_obs]

    old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
    old_t = np.zeros(n_boxes)  # Track last push time per box

    for i, (t, box) in enumerate(push_times_and_box):
        if t == old_t[int(box)]:
            continue  # Skip redundant updates if push time is unchanged for the same box

        belief_avail_event[i] = np.array([
            posterior.boxes[j].marginalize(
                obs_supp_args=[t - old_t[j]], latent_kwargs={'record': old_idx[j]}
            )
            for j in range(n_boxes)
        ])

        old_idx[int(box)] += 1  # Update observation index for the box
        old_t[int(box)] = t  # Update last push time for the box

    return belief_avail_event

@process_block_safely
def compute_reward_probabilities(
        df: pd.DataFrame,
        index: tuple
) -> np.ndarray[float]:
    """
    Compute the exact reward probability of each right before each push.

    Args:
        df: Pandas DataFrame containing session data.
        index: Index to locate the relevant block data.

    Returns:
        np.ndarray: An event-based array of shape (n_obs, n_boxes),
                    where reward probabilities are evaluated before each push.
    """

    df_block = df.loc[index]
    schedules = -np.sort(-df_block['schedule'].unique())
    n_boxes = len(schedules)
    shape = df_block.index.unique('shape')[0]  # Assume agent knows number of states perfectly

    # Construct likelihood/observation model
    obs_model = models.GammaObservation(shape)

    # Compute availability marginal for each push
    n_obs = df_block['push times'].size - np.count_nonzero(np.isnan(df_block['push times']))
    belief_avail_event = np.zeros((n_obs, n_boxes))
    push_times_and_box = df_block[['push times', 'box rank']].values[:n_obs]
    old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
    old_t = np.zeros(n_boxes)  # Track last push time per box

    for i, (t, box) in enumerate(push_times_and_box):
        if t == old_t[int(box)]:
            continue  # Skip redundant updates if push time is unchanged for the same box

        belief_avail_event[i] = np.array([
            obs_model.probability((1, t - old_t[j]), schedules[j])
            for j in range(n_boxes)
        ])

        old_idx[int(box)] += 1  # Update observation index for the box
        old_t[int(box)] = t  # Update last push time for the box

    return belief_avail_event


@process_block_safely
def compute_joint_beliefs(
        df: pd.DataFrame,
        index: tuple,
        posterior: models.Posterior
) -> np.ndarray:
    """
    Computes the joint belief of the reward availability and schedule of each box for each push event.

    Args:
        df: Pandas DataFrame containing session data with columns 'push times' and 'box rank'.
        index: Index or identifier to select a specific block from the DataFrame.
        posterior: An instance of the Posterior class that contains the belief states and methods
                  for updating and computing beliefs. This posterior object is used to compute
                  joint beliefs for the reward availability of each box.

    Returns:
        np.ndarray: A 4D array representing the joint beliefs for each observation, with shape
                    (n_obs, n_boxes, 2, len(posterior.support())).
                    The first dimension represents the observation index (push event),
                    the second dimension represents each box,
                    the third dimension represents the belief for reward availability (binary: 0/1),
                    and the fourth dimension corresponds to the support of the posterior distribution.
    """

    n_boxes = posterior.n_boxes
    block_data = df.loc[index]

    # Compute availability marginal for each push
    n_obs = block_data['push times'].size - np.count_nonzero(np.isnan(block_data['push times']))
    belief_joint_event = np.zeros((n_obs, n_boxes, 2, len(posterior.support())))
    push_times_and_box = block_data[['push times', 'box rank']].values[:n_obs]
    old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
    old_t = np.zeros(n_boxes)  # Track last push time per box

    for i, (t, box) in enumerate(push_times_and_box):
        if t == old_t[int(box)]:
            continue  # Skip redundant updates if push time is unchanged for the same box

        belief_joint_event[i] = np.array([
            posterior.boxes[j].joint(obs_supp_args=[t - old_t[j]], latent_kwargs={'record': old_idx[j]})
            for j in range(n_boxes)
        ])

        old_idx[int(box)] += 1  # Update observation index for the box
        old_t[int(box)] = t  # Update last push time for the box

    return belief_joint_event

def predict_pushed_box(df: pd.DataFrame, x: np.ndarray, col_name: str = 'box rank') -> tuple[float, Any]:
    """
    Predicts the pushed box using multinomial logistic regression and evaluates the accuracy of predictions.

    Args:
        df: pandas DataFrame containing session data, including a column with the target labels (box rank).
        x: A 2D numpy array containing the features (independent variables) used for prediction.
        col_name: The name of the column in `df` that contains the target labels (default is 'box rank').

    Returns:
        A tuple containing:
            - A float representing the accuracy of the predictions (mean of correct predictions).
            - The fitted multinomial logistic regression model (`MNLogitResults` object).
    """
    y = df[col_name]
    mdl = smf.mnlogit("y ~ X", {'y': y, 'X': x}).fit()
    yhat = np.argmax(mdl.predict(), axis=1)
    accuracy = (yhat == y).mean()
    return accuracy, mdl

def get_mean_beliefs(beliefs: ArrayLike | list, supp: ArrayLike | list) -> ArrayLike | list:
    """
    Computes the mean beliefs over time.

    Args:
        beliefs: Matrix or list of beliefs for each box over time.
        supp: The support for the beliefs (values over which the beliefs are computed).

    Returns:
        A vector of belief means over time.
    """
    if isinstance(beliefs, np.ndarray):
        return (beliefs @ supp[:, np.newaxis]).squeeze()  # E[lambda] at each timepoint
    return [x @ supp for x in beliefs]

def get_std_beliefs(beliefs: ArrayLike | list, supp: ArrayLike | list) -> np.ndarray | list:
    """
    Computes the std of beliefs over time.

    Args:
        beliefs: Matrix or list of beliefs for each box over time.
        supp: Support of the beliefs.

    Returns:
        A vector of the std of beliefs over time.
    """
    if isinstance(beliefs, np.ndarray):
        return np.sqrt(beliefs @ (supp[:, np.newaxis] ** 2) - (beliefs @ supp[:, np.newaxis]) ** 2).squeeze()
    return [np.sqrt(x @ (supp**2) - (x @ supp)**2) for x in beliefs]
