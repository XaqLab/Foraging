"""
This module contains functions for computing beliefs.
"""

import logging
from copy import deepcopy
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import ArrayLike
from scipy.stats import entropy, gamma, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from foraging.models._base import HashableDict
from foraging.models.deep import PolicyMLP, ActionMLP
from foraging.models.distribution import (
    FactorizedPosterior,
    IndexedObservation,
    Posterior,
    Probabilities,
    RewardInterval,
    RewardOutcome,
    SupportsUpdatesByBox,
)
from foraging.models.experiment import Experiment
from foraging.utils.data import map_box_positions_to_ranks

# from foraging.utils.stats import mcfadden_pseudo_rsquared, permutation_test_logistic

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Calculate fisher information
def fisher_info_reward_observations(
    t: float | ArrayLike, schedule: float, alpha: float
) -> float:
    """Calculate the Fisher information for reward observations.

    Args:
        t: Time(s) since last push of the observation.
        schedule: Reward schedule of the observation.
        alpha: Alpha parameter of the Gamma distribution.

    Returns:
        Fisher information for the reward observations.
    """
    scale = schedule / alpha
    cdf = gamma.cdf(t, a=alpha, scale=scale)

    # Handle vectorized operations with explicit warning suppression
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = cdf * (1 - cdf)
        result = (
            (t / schedule) ** 2 * gamma.pdf(t, a=alpha, scale=scale) ** 2 / denominator
        )

        # Replace invalid values (inf, nan) with 0
        if np.isscalar(result):
            return 0.0 if not np.isfinite(result) else result
        else:
            result = np.where(np.isfinite(result), result, 0.0)
            return result


def _create_observation(
    box: int, is_available: bool, time: float
) -> IndexedObservation[RewardOutcome]:
    """
    Factory function that creates an IndexedObservation with RewardObservation.

    Args:
        box: Box position/index
        is_available: Whether reward is available
        time: Time value for the observation

    Returns:
        IndexedObservation[RewardObservation] that can be used anywhere an IndexedObservation is expected
    """
    return IndexedObservation(
        i=box, observation=RewardOutcome(is_available=is_available, time=time)
    )


def _create_perfect_observation(
    box: int, time: float
) -> IndexedObservation[RewardOutcome]:
    """
    Factory function that creates an IndexedObservation with RewardObservation.

    Args:
        box: Box position/index
        is_available: Whether reward is available
        time: Time value for the observation

    Returns:
        IndexedObservation[RewardObservation] that can be used anywhere an IndexedObservation is expected
    """
    return IndexedObservation(i=box, observation=RewardInterval(time=time))


def compute_schedule_posterior(
    dataset: Experiment,
    block_key: dict[str, Any],
    block: pd.DataFrame,
    posterior_maker: Callable[
        [Experiment, dict[str, Any], pd.DataFrame], SupportsUpdatesByBox
    ],
    *args,
    **kwargs,
) -> dict[HashableDict, FactorizedPosterior]:
    """
    Computes the posterior belief over reward schedules for each box, updating after each push.

    Args:
        dataset: Experiment containing session data.
        block_key: Key for the block.
        block: DataFrame containing block data.
        posterior_maker: Function that instantiates the posterior.
        *args, **kwargs: Additional arguments to pass to the posterior_maker.
    Returns:
        dict[HashableDict, FactorizedPosterior]: the computed beliefs about reward schedules.
    """
    # Iterate over each box in order from fastest to slowest
    n_obs = len(block)
    push_times = block["push times"].values
    push_intervals = block["push intervals"].values
    reward_outcomes = block["reward outcomes"].values
    box_positions = block["box position"].values.astype(int)

    # Construct posterior
    beliefs = posterior_maker(dataset, block_key, block, *args, **kwargs)
    for i in range(n_obs):
        beliefs.update(
            block_key.update("push time", push_times[i]),
            _create_observation(
                box=box_positions[i],
                is_available=reward_outcomes[i],
                time=push_intervals[i],
            ),
        )
    return beliefs


def compute_joint_schedule_reward_posterior(
    dataset: Experiment,
    block_key: dict[str, Any],
    block: pd.DataFrame,
    posterior_maker: Callable[
        [Experiment, dict[str, Any], pd.DataFrame], SupportsUpdatesByBox
    ],
    *args,
    **kwargs,
) -> dict[HashableDict, np.ndarray]:
    """
    Computes the joint posterior over schedule permutation and reward availability at each box.

    For permutation-based beliefs, this computes P(permutation, reward_available | observations)
    by marginalizing over the reward CDF given each permutation hypothesis.

    Args:
        dataset: Experiment containing session data.
        block_key: Key for the block.
        block: DataFrame containing block data.
        posterior_maker: Function that instantiates the posterior (should be permutation-based).
        *args, **kwargs: Additional arguments to pass to the posterior_maker.

    Returns:
        dict[HashableDict, np.ndarray]: Dictionary mapping timestep keys to 2D arrays of shape
            (n_permutations, n_boxes) where entry [i, j] is P(permutation_i, reward_available_at_box_j).
    """

    n_obs = len(block)
    push_times = block["push times"].values
    push_intervals = block["push intervals"].values
    reward_outcomes = block["reward outcomes"].values
    box_positions = block["box position"].values.astype(int)

    # Get unique box positions and schedule info
    n_boxes = block["box position"].nunique()
    assigned_schedules = block.index.get_level_values("assigned schedules")[0]
    shape = (
        block.index.get_level_values("shape")[0]
        if "shape" in block.index.names
        else block["shape"].iloc[0]
    )

    # Construct permutation posterior over schedules
    beliefs = posterior_maker(dataset, block_key, block, *args, **kwargs)
    permutation_hypotheses = beliefs.prior.support
    n_perms = len(permutation_hypotheses)
    last_push_times = np.zeros(n_boxes)

    # Compute joint distributions at each timestep
    def _update_joint(
        schedule_belief: Probabilities,
        push_time: float,
        last_push_times: Iterable[float],
    ):
        joint = np.zeros((n_perms, n_boxes))
        for i, perm_hyp in enumerate(permutation_hypotheses):
            for box in range(n_boxes):
                schedule = perm_hyp.schedule[box]
                p_available = gamma.cdf(
                    push_time - last_push_times[box], shape, scale=schedule / shape
                )
                joint[i, box] = schedule_belief.query_by_index(i) * p_available
        return joint

    # Handle prior
    key = block_key.update("push time", 0)
    joint_posterior = {key: _update_joint(beliefs.prior, 0, last_push_times)}
    for i in range(n_obs):
        # Get the beliefs right before the push
        old_key = key
        key = block_key.update("push time", push_times[i])
        joint_posterior[key] = _update_joint(
            beliefs[old_key], push_times[i], last_push_times
        )
        last_push_times[box_positions[i]] = push_times[i]

        # Update the beliefs after the push
        beliefs.update(
            key,
            _create_observation(
                box=box_positions[i],
                is_available=reward_outcomes[i],
                time=push_intervals[i],
            ),
        )
    return joint_posterior


def compute_joint_schedule_reward_posterior_discrete_time(
    dataset: Experiment,
    block_key: dict[str, Any],
    block: pd.DataFrame,
    posterior_maker: Callable[
        [Experiment, dict[str, Any], pd.DataFrame], SupportsUpdatesByBox
    ],
    dt: float = 1.0 ,
    *args,
    **kwargs,
) -> dict[HashableDict, np.ndarray]:
    """
    Computes the joint posterior over schedule permutation and reward availability at each box
    at discrete time steps of size dt between pushes.

    For permutation-based beliefs, this computes P(permutation, reward_available | observations)
    by marginalizing over the reward CDF given each permutation hypothesis. Beliefs are
    evaluated at times 0, dt, 2*dt, ... up to the last push time in the block.

    Args:
        dataset: Experiment containing session data.
        block_key: Key for the block.
        block: DataFrame containing block data.
        posterior_maker: Function that instantiates the posterior (should be permutation-based).
        dt: Size of discrete time steps. Joint posterior is computed at t = 0, dt, 2*dt, ...
        *args, **kwargs: Additional arguments to pass to the posterior_maker.

    Returns:
        dict[HashableDict, np.ndarray]: Dictionary mapping time keys to 2D arrays of shape
            (n_permutations, n_boxes) where entry [i, j] is P(permutation_i, reward_available_at_box_j).
            Keys use block_key.update("push time", t) for each discrete t.
    """

    n_obs = len(block)
    push_times = block["push times"].values
    push_intervals = block["push intervals"].values
    reward_outcomes = block["reward outcomes"].values
    box_positions = block["box position"].values.astype(int)

    # Get unique box positions and schedule info
    n_boxes = block["box position"].nunique()
    shape = (
        block.index.get_level_values("shape")[0]
        if "shape" in block.index.names
        else block["shape"].iloc[0]
    )

    # Construct permutation posterior over schedules
    beliefs = posterior_maker(dataset, block_key, block, *args, **kwargs)
    permutation_hypotheses = beliefs.prior.support
    n_perms = len(permutation_hypotheses)

    # Compute joint at a given time
    def _update_joint(
        schedule_belief: Probabilities,
        t: float,
        last_push_times: np.ndarray,
    ):
        joint = np.zeros((n_perms, n_boxes))
        for i, perm_hyp in enumerate(permutation_hypotheses):
            for box in range(n_boxes):
                schedule = perm_hyp.schedule[box]
                p_available = gamma.cdf(
                    t - last_push_times[box], shape, scale=schedule / shape
                )
                joint[i, box] = schedule_belief.query_by_index(i) * p_available
        return joint

    # Pass 1: update beliefs at each push and record last_push_times after each push
    last_push_times = np.zeros(n_boxes)
    # last_push_times_after_push[k] = last_push_times after push k (so before push k+1)
    last_push_times_after_push = [last_push_times.copy()]

    key = block_key.update("push time", 0)
    for i in range(n_obs):
        old_key = key
        key = block_key.update("push time", push_times[i])
        if i == 0:
            belief_state = beliefs.prior
        else:
            belief_state = beliefs[old_key]
        beliefs.update(
            key,
            _create_observation(
                box=box_positions[i],
                is_available=reward_outcomes[i],
                time=push_intervals[i],
            ),
        )
        last_push_times[box_positions[i]] = push_times[i]
        last_push_times_after_push.append(last_push_times.copy())

    # Pass 2: discrete time points 0, dt, 2*dt, ... up to last push time (include push times)
    t_max = float(push_times[-1]) if n_obs > 0 else 0.0
    # Grid: 0, dt, 2*dt, ... up to t_max; merge with push_times so exact push times included
    grid_times = np.arange(0, t_max + dt * 0.5, dt)
    discrete_times = np.unique(np.concatenate([grid_times, push_times]))
    discrete_times = discrete_times[discrete_times <= t_max]

    joint_posterior = {}
    for t in discrete_times:
        # Use beliefs and last_push_times from after the last push at or before t
        if n_obs == 0 or t < push_times[0]:
            belief_state = beliefs.prior
            lpt = np.zeros(n_boxes)
        else:
            k = np.searchsorted(push_times, t, side="right") - 1  # last push index with push_times[k] <= t
            key_at_t = block_key.update("push time", push_times[k])
            belief_state = beliefs[key_at_t]
            lpt = last_push_times_after_push[k + 1]

        time_key = block_key.update("push time", t)
        joint_posterior[time_key] = _update_joint(belief_state, t, lpt)

    return joint_posterior



def fit_policy_to_predict_future_behavior(
    dataset: Experiment,
    beliefs: dict[HashableDict, Any],
    conds: dict[str, Any] = None,
    split_frac: Iterable[float] = [0.8, 0.2],
    hidden_dims: Iterable[int] = [64, 32],
    learning_rate: float = 0.001,
    batch_size: int = 32,
    n_epochs: int = 100,
    patience: int = 10,
    seed: int = 42,
    tensorboard_dir: str = None,
    device: str = None,
    *args,
    **kwargs,
):
    """
    Fit a policy (MLP) to predict future behavior from beliefs.

    Args:
        dataset: Experiment containing session data.
        beliefs: Dictionary mapping block keys to posterior objects containing beliefs at each push time.
        conds: Optional conditions to filter the dataset.
        split_frac: Train/test split fractions (must sum to 1.0).
        hidden_dims: List of hidden layer dimensions for the MLP.
        learning_rate: Learning rate for optimizer.
        batch_size: Batch size for training.
        n_epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        seed: Random seed for reproducibility.
        tensorboard_dir: Optional directory for tensorboard logging.
        device: Device to train on ('cuda' or 'cpu'). If None, auto-detects.

    Returns:
        Trained PyTorch model.
    """

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # 1. Filter dataset
    dataset = dataset.filter(conds)

    # 2. Format data into supervised learning format
    X_list = []  # Belief vectors
    y_interval_list = []  # Push intervals (regression target)
    y_box_list = []  # Box choices (classification target)

    # Get beliefs of each block using the standard block-processing pattern
    def _inner(
        dataset: "Experiment",
        block_key: dict[str, Any],
        block: pd.DataFrame,
        *args,
        **kwargs,
    ):
        # Work on a copy with a simple RangeIndex
        block = block.reset_index()

        # Look up the posterior corresponding to this block
        if block_key not in beliefs:
            return None
        posterior = beliefs[block_key]

        # Extract push times and iterate through the posterior
        n_obs = len(block)
        push_times = block["push times"].values
        chosen_boxes = block["box position"].values.astype(int)
        push_intervals = block["push intervals"].values

        # For each push (except the last), use current belief to predict next action
        for i in range(n_obs):
            pt = push_times[i]
            key = block_key.update("push time", pt)

            if key not in posterior:
                continue

            # Get belief at this timestep
            belief = posterior[key]

            # Extract belief representation (flatten if needed)
            if hasattr(belief, "representation"):
                belief_vec = np.asarray(belief.representation)
            else:
                belief_vec = np.asarray(belief)

            # Flatten belief vector
            belief_vec = belief_vec.flatten()

            # Get next action (push interval and box choice)
            next_interval = push_intervals[i]
            next_box = chosen_boxes[i]

            X_list.append(belief_vec)
            y_interval_list.append(next_interval)
            y_box_list.append(next_box)

        # All supervision is accumulated via the outer lists; nothing to return.
        return None

    # Walk all blocks once, accumulating supervised examples
    dataset.process_blocks(_inner)

    if len(X_list) == 0:
        raise ValueError("No data available after filtering and formatting.")

    # Convert to numpy arrays
    X = np.array(X_list)
    y_interval = np.array(y_interval_list)
    y_box = np.array(y_box_list)

    # Get dimensions
    input_dim = X.shape[1]
    n_boxes = len(np.unique(y_box))

    logger.info(
        f"Formatted {len(X)} samples with input_dim={input_dim}, n_boxes={n_boxes}"
    )

    # 3. Train/test split
    assert np.isclose(sum(split_frac), 1.0), "split_frac must sum to 1.0"
    n_samples = len(X)
    n_train = int(n_samples * split_frac[0])

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_interval_train, y_interval_test = y_interval[train_idx], y_interval[test_idx]
    y_box_train, y_box_test = y_box[train_idx], y_box[test_idx]

    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_interval_train_t = torch.FloatTensor(y_interval_train).unsqueeze(1).to(device)
    y_interval_test_t = torch.FloatTensor(y_interval_test).unsqueeze(1).to(device)
    y_box_train_t = torch.LongTensor(y_box_train).to(device)
    y_box_test_t = torch.LongTensor(y_box_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_interval_train_t, y_box_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Instantiate MLP model (class defined at module level)
    model = PolicyMLP(input_dim, hidden_dims, n_boxes).to(device)

    # Define loss functions and optimizer
    criterion_interval = nn.MSELoss()
    criterion_box = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup tensorboard if requested
    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(tensorboard_dir)

    # Training loop
    best_test_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    logger.info(f"Training on device: {device}")
    for epoch in range(n_epochs):
        model.train()
        train_loss_interval = 0.0
        train_loss_box = 0.0

        for batch_X, batch_y_interval, batch_y_box in train_loader:
            optimizer.zero_grad()

            # Forward pass
            interval_pred, box_logits = model(batch_X)

            # Compute losses
            loss_interval = criterion_interval(interval_pred, batch_y_interval)
            loss_box = criterion_box(box_logits, batch_y_box)
            loss = loss_interval + loss_box

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_interval += loss_interval.item()
            train_loss_box += loss_box.item()

        train_loss_interval /= len(train_loader)
        train_loss_box /= len(train_loader)
        train_loss = train_loss_interval + train_loss_box

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            interval_pred_test, box_logits_test = model(X_test_t)
            test_loss_interval = criterion_interval(
                interval_pred_test, y_interval_test_t
            ).item()
            test_loss_box = criterion_box(box_logits_test, y_box_test_t).item()
            test_loss = test_loss_interval + test_loss_box

            # Classification accuracy
            box_pred_test = torch.argmax(box_logits_test, dim=1)
            test_acc_box = (box_pred_test == y_box_test_t).float().mean().item()

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/train_total", train_loss, epoch)
            writer.add_scalar("Loss/train_interval", train_loss_interval, epoch)
            writer.add_scalar("Loss/train_box", train_loss_box, epoch)
            writer.add_scalar("Loss/test_total", test_loss, epoch)
            writer.add_scalar("Loss/test_interval", test_loss_interval, epoch)
            writer.add_scalar("Loss/test_box", test_loss_box, epoch)
            writer.add_scalar("Accuracy/test_box", test_acc_box, epoch)

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}/{n_epochs}: "
                f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
                f"Test Box Acc={test_acc_box:.4f}"
            )

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if writer is not None:
        writer.close()

    logger.info(f"Training complete. Best test loss: {best_test_loss:.4f}")

    return model


def fit_policy_to_predict_future_behavior_v2(
    dataset: Experiment,
    beliefs: dict[HashableDict, Any],
    conds: dict[str, Any] = None,
    split_frac: Iterable[float] = [0.8, 0.2],
    hidden_dims: Iterable[int] = [64, 32],
    learning_rate: float = 0.001,
    batch_size: int = 32,
    n_epochs: int = 100,
    patience: int = 10,
    seed: int = 42,
    tensorboard_dir: str = None,
    device: str = None,
    bin_size: float = 1.0,
    max_bins: int = None,
    permute_labels: bool = False,
    permute_seed: int = None,
    *args,
    **kwargs,
):
    """
    Fit a policy (MLP) to predict the next time-step's action from the current time-step's beliefs.

    Beliefs are assumed to be defined over discrete time steps (e.g. from
    compute_joint_schedule_reward_posterior_discrete_time with the same bin_size).
    For each consecutive pair of discrete times (t_i, t_{i+1}), we use the belief at t_i
    to predict the action at t_{i+1}. The action at a time step is: push fast/medium/slow
    box (0/1/2) if a push occurs in that bin, else wait (3).

    Action encoding:
    - Action 0: push fast box (box 0)
    - Action 1: push medium box (box 1)
    - Action 2: push slow box (box 2)
    - Action 3: wait
    n_actions = n_boxes + 1.

    Args:
        dataset: Experiment containing session data.
        beliefs: Dict mapping block keys to posteriors keyed by discrete time (e.g. keys
            like block_key.update("push time", t) for t = 0, bin_size, 2*bin_size, ...).
        conds: Optional conditions to filter the dataset.
        split_frac: Train/test split fractions (must sum to 1.0).
        hidden_dims: List of hidden layer dimensions for the MLP.
        learning_rate: Learning rate for optimizer.
        batch_size: Batch size for training.
        n_epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        seed: Random seed for reproducibility.
        tensorboard_dir: Optional directory for tensorboard logging.
        device: Device to train on ('cuda' or 'cpu'). If None, auto-detects.
        bin_size: Size of each time bin for discretizing push intervals.
        max_bins: Not currently used (kept for API compatibility). Action space is fixed at n_boxes + 1.
        permute_labels: If True, randomly permute action labels before training (useful for baselines).
        permute_seed: Random seed for label permutation. If None, uses the main seed.

    Returns:
        Trained PyTorch model.
    """

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # 1. Filter dataset
    dataset = dataset.filter(conds)

    # 2. First pass: collect unique boxes to determine n_boxes
    n_boxes = len(dataset.get_unique("box rank"))

    # 3. Format data into supervised learning format
    # Predict NEXT time-step's action from CURRENT time-step's beliefs.
    # Beliefs are defined over discrete time steps (0, bin_size, 2*bin_size, ...).
    X_list = []  # Belief vectors at current time step
    y_action_list = []  # Action labels at NEXT time step (classification target)

    # Action encoding:
    # - Action 0: push fast box (box 0)
    # - Action 1: push medium box (box 1)
    # - Action 2: push slow box (box 2)
    # - Action 3: wait
    n_actions = n_boxes + 1

    # Get beliefs of each block; use current-step belief to predict next-step action
    def _inner(
        dataset: "Experiment",
        block_key: dict[str, Any],
        block: pd.DataFrame,
        *args,
        **kwargs,
    ):
        block = block.reset_index()
        if block_key not in beliefs:
            return None
        posterior = beliefs[block_key]

        # Discrete times: keys in posterior are (block_key + "push time", t); get sorted t
        try:
            time_keys = list(posterior.keys())
        except AttributeError:
            return None
        discrete_times = sorted(set(k["push time"] for k in time_keys if "push time" in k))
        if len(discrete_times) < 2:
            return None

        # Build action-at-time from block pushes: bin k has a push -> action = box; else wait
        push_times_arr = block["push times"].values
        chosen_boxes = block["box position"].values.astype(int)
        action_at_bin = {}
        for i in range(len(push_times_arr)):
            pt = push_times_arr[i]
            bin_idx = int(np.floor(pt / bin_size))
            action_at_bin[bin_idx] = chosen_boxes[i]

        def action_at_time(t: float):
            bin_idx = int(np.floor(t / bin_size))
            return action_at_bin.get(bin_idx, n_boxes)

        # For each consecutive pair (t_i, t_{i+1}): X = belief at t_i, y = action at t_{i+1}
        for i in range(len(discrete_times) - 1):
            t_curr = discrete_times[i]
            t_next = discrete_times[i + 1]
            key_curr = block_key.update("push time", t_curr)
            if key_curr not in posterior:
                continue
            belief = posterior[key_curr]
            if hasattr(belief, "representation"):
                belief_vec = np.asarray(belief.representation)
            else:
                belief_vec = np.asarray(belief)
            belief_vec = belief_vec.flatten()

            action_next = action_at_time(t_next)
            X_list.append(belief_vec)
            y_action_list.append(action_next)

        return None

    dataset.process_blocks(_inner)

    if len(X_list) == 0:
        raise ValueError("No data available after filtering and formatting.")

    # Convert to numpy arrays
    X = np.array(X_list)
    y_action = np.array(y_action_list)

    # Get dimensions
    input_dim = X.shape[1]
    
    # Action space is fixed: n_actions = n_boxes + 1
    # - Actions 0 to (n_boxes-1): push box 0 to (n_boxes-1)
    # - Action n_boxes: wait
    # n_actions is already set above in the _inner function scope, but we need it here too
    n_actions = n_boxes + 1
    n_samples = len(X)
    
    # Verify that all action labels are valid
    max_action = max(y_action) if n_samples > 0 else 0
    if max_action >= n_actions:
        raise ValueError(
            f"Invalid action {max_action} found. Expected actions in range [0, {n_actions-1}]. "
            f"n_boxes={n_boxes}, so n_actions={n_actions}."
        )

    # Apply label permutation if requested (remap action IDs, keep X-y pairing)
    if permute_labels:
        perm_seed = permute_seed if permute_seed is not None else seed
        rng_perm = np.random.RandomState(perm_seed)
        y_action = y_action[rng_perm.permutation(n_samples)]
        logger.info(
            f"Permuted action labels using seed {perm_seed}. "
        )
    else:
        logger.info(
            f"Formatted {len(X)} samples with input_dim={input_dim}, n_boxes={n_boxes}, "
            f"n_actions={n_actions}, bin_size={bin_size}"
        )

    # 3. Train/test split
    assert np.isclose(sum(split_frac), 1.0), "split_frac must sum to 1.0"
    n_samples = len(X)
    n_train = int(n_samples * split_frac[0])

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_action_train, y_action_test = y_action[train_idx], y_action[test_idx]

    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_action_train_t = torch.LongTensor(y_action_train).to(device)
    y_action_test_t = torch.LongTensor(y_action_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_action_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Create a simple classification MLP (single head for action prediction)   
    model = ActionMLP(input_dim, hidden_dims, n_actions).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup tensorboard if requested
    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(tensorboard_dir)

    # Training loop
    best_test_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    logger.info(f"Training on device: {device}")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y_action in train_loader:
            optimizer.zero_grad()

            # Forward pass
            action_logits = model(batch_X)

            # Compute loss
            loss = criterion(action_logits, batch_y_action)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            action_logits_test = model(X_test_t)
            test_loss = criterion(action_logits_test, y_action_test_t).item()

            # Classification accuracy
            action_pred_test = torch.argmax(action_logits_test, dim=1)
            test_acc = (action_pred_test == y_action_test_t).float().mean().item()

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}/{n_epochs}: "
                f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
                f"Test Acc={test_acc:.4f}"
            )

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if writer is not None:
        writer.close()

    logger.info(f"Training complete. Best test loss: {best_test_loss:.4f}")

    return model


def compute_perfect_obs_schedule_posterior(
    dataset: Experiment,
    block_key: dict[str, Any],
    block: pd.DataFrame,
    posterior_maker: Callable[
        [Experiment, dict[str, Any], pd.DataFrame], SupportsUpdatesByBox
    ],
    *args,
    **kwargs,
) -> dict[HashableDict, FactorizedPosterior]:
    """
    Computes the posterior belief over reward schedules for each box, updating after each push.

    Args:
        dataset: Experiment containing session data.
        block_key: Key for the block.
        block: DataFrame containing block data.
        posterior_maker: Function that instantiates the posterior.
        *args, **kwargs: Additional arguments to pass to the posterior_maker.
    Returns:
        dict[HashableDict, FactorizedPosterior]: the computed beliefs about reward schedules.
    """
    # Iterate over each box in order from fastest to slowest
    n_obs = len(block)
    push_times = block["push times"].values
    push_intervals = block["push intervals"].values
    reward_intervals = block["reward intervals"].values
    box_positions = block["box position"].values.astype(int)

    # Construct posterior
    beliefs = posterior_maker(dataset, block_key, block, *args, **kwargs)
    for i in range(n_obs):
        beliefs.update(
            block_key.update("push time", push_times[i]),
            _create_perfect_observation(
                box=box_positions[i],
                time=reward_intervals[i],
            ),
        )
    return beliefs


def compute_accuracy(
    dataset: Experiment,
    block_key: dict[str, Any],
    block: pd.DataFrame,
    beliefs: dict[HashableDict, Posterior],
    *args,
    seed: int = 42,
    n_samples: int = 100,
    n_correct: int = -1,
    **kwargs,
) -> ArrayLike:
    """
    Computes the accuracy of the beliefs over reward schedules for each box, updating after each push.

    Args:
        dataset: Experiment containing session data.
        block_key: Key for the block.
        block: DataFrame containing block data.
        beliefs: dict[HashableDict, Posterior] containing the schedule beliefs.
        seed: Random seed for reproducibility.
        n_samples: Number of samples to draw for Monte Carlo estimation.
        n_correct: Number of correct pairwise order relations required for counting a sample as "correct".
            If -1, all pairwise relations must be correct (i.e. the sampled total order matches the true order).
    Returns:
        ArrayLike: the computed accuracy of the beliefs over reward schedules.
    """
    # Extract data corresponding to the given index
    box_ranks = map_box_positions_to_ranks(block)
    belief_block = beliefs[block_key]
    n_boxes = len(box_ranks)
    if n_correct == -1:
        n_correct = (n_boxes * (n_boxes - 1)) // 2
    n_steps = len(belief_block)
    push_times = block["push times"].values
    samples = np.zeros((n_steps, n_boxes, n_samples))

    rng = np.random.default_rng(seed + int(block_key["block"]))
    samples[0] = belief_block[block_key.update("push time", 0)].sample(n_samples, rng)

    # Estimate accuracy via Monte Carlo sampling
    for i, pt in enumerate(push_times):
        belief = belief_block[block_key.update("push time", pt)]
        samples[i + 1] = belief.sample(n_samples, rng)

    # Count the fraction of samples that match the true order
    sampled_orders = np.argsort(np.argsort(samples, axis=1), axis=1)
    true_order = box_ranks["box rank"].values

    # Count correct pairwise order relations per sample, and compute fraction meeting >= n_correct.
    true_rel = true_order[:, None] < true_order[None, :]  # (boxes, boxes)
    sampled_rel = (
        sampled_orders[:, :, None, :] < sampled_orders[:, None, :, :]
    )  # (steps, boxes, boxes, samples)

    pair_mask = np.triu(np.ones(true_rel.shape, dtype=bool), 1)  # (boxes, boxes)

    # count correct relations per (step, sample)
    correct_counts = (
        (sampled_rel == true_rel[None, :, :, None]) & pair_mask[None, :, :, None]
    ).sum(
        axis=(1, 2)
    )  # (steps, samples)
    accuracies = (correct_counts >= n_correct).mean(axis=1)
    return accuracies


def compute_when_accuracy_threshold_met(
    dataset: Experiment,
    block_key: dict[str, Any],
    block: pd.DataFrame,
    beliefs: dict[HashableDict, Posterior],
    *args,
    threshold: float = 0.8,
    seed: int = 42,
    n_samples: int = 100,
    n_correct: int = -1,
    n_threshold_pts: int = 10,
    **kwargs,
) -> float:
    """
    Find the earliest time-step when accuracy meets/exceeds `threshold` and stays there for the rest of the block.

    This is intended for permutation-based beliefs where samples correspond to schedule values per box.

    Args:
        dataset: Experiment containing session data.
        block_key: Key for the block.
        block: DataFrame containing block data.
        beliefs: dict[HashableDict, Posterior] containing the schedule beliefs.
        threshold: Accuracy threshold in [0, 1]. Default is 0.8.
        seed: Random seed for reproducibility (passed to `compute_accuracy`).
        n_samples: Number of samples to draw for Monte Carlo estimation.
        n_correct: Number of correct pairwise order relations required for counting a sample as "correct".
            If -1, all pairwise relations must be correct (i.e. the sampled total order matches the true order).
        n_threshold_pts: Number of points to check for threshold met. If -1, check all points.
        **kwargs: Unused; kept for API compatibility with other block processors.

    Returns:
        The push time (including 0 for the prior) where the threshold is first met and never falls below afterwards.
        If no such time exists, returns -1.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    accuracies = compute_accuracy(
        dataset=dataset,
        block_key=block_key,
        block=block,
        beliefs=beliefs,
        seed=seed,
        n_samples=n_samples,
        n_correct=n_correct,
        **kwargs,
    )

    times = np.insert(block["push times"].values, 0, 0)
    # Find earliest i such that accuracies[i:] are all >= threshold.
    for i in range(len(accuracies)):
        # if np.all(accuracies[i:] >= threshold):
        #     return float(times[i])
        if n_threshold_pts == -1 and np.all(accuracies[i:] >= threshold):
            return float(times[i])
        elif n_threshold_pts > 0 and i + n_threshold_pts <= len(accuracies):
            if np.all(accuracies[i : i + n_threshold_pts] >= threshold):
                return float(times[i])
    return -1


def compute_per_box_accuracy(
    dataset: Experiment,
    block_key: dict[str, Any],
    block: pd.DataFrame,
    beliefs: dict[HashableDict, Posterior],
    *args,
    seed: int = 42,
    n_samples: int = 100,
    **kwargs,
) -> ArrayLike:
    """
    Computes the accuracy of the beliefs over reward schedules for each box, updating after each push.

    Args:
        dataset: Experiment containing session data.
        block_key: Key for the block.
        block: DataFrame containing block data.
        beliefs: dict[HashableDict, Posterior] containing the schedule beliefs.
        seed: Random seed for reproducibility.
        n_samples: Number of samples to draw for Monte Carlo estimation.
        n_correct: Number of correct relations to require for accuracy calculation. If -1, all relations must be correct.
    Returns:
        ArrayLike: the computed accuracy of the beliefs over reward schedules.
    """
    # Extract data corresponding to the given index
    box_ranks = map_box_positions_to_ranks(block)
    belief_block = beliefs[block_key]
    n_boxes = len(box_ranks)
    n_steps = len(belief_block)
    push_times = block["push times"].values
    samples = np.zeros((n_steps, n_boxes, n_samples))
    rng = np.random.default_rng(seed + int(block_key["block"]))
    samples[0] = belief_block[block_key.update("push time", 0)].sample(n_samples, rng)

    # sort fast > medium > slow
    sorted_idx = np.argsort(block.index.unique("assigned schedules")[0])

    # Estimate accuracy via Monte Carlo sampling
    for i, pt in enumerate(push_times):
        belief = belief_block[block_key.update("push time", pt)]
        samples[i + 1] = belief.sample(n_samples, rng)

    # Count the fraction of samples that match the true order
    sampled_orders = np.argsort(np.argsort(samples, axis=1), axis=1)
    sampled_orders = sampled_orders[:, sorted_idx, :]
    true_order = box_ranks["box rank"].values[sorted_idx]

    # Count correct pairwise order relations per sample, and compute fraction meeting >= n_correct.
    true_rel = true_order[:, None] < true_order[None, :]  # (boxes, boxes)
    sampled_rel = (sampled_orders[:, :, None, :] < sampled_orders[:, None, :, :]).mean(
        axis=3
    )  # (steps, boxes, boxes, samples)
    return {"correct_counts": sampled_rel, "time": np.insert(push_times, 0, 0)}
    # return (
    #     (sampled_orders == true_order[np.newaxis, :, np.newaxis])
    #     .all(axis=1)
    #     .mean(axis=1)
    # ) # cute one-liner


# @process_block_safely
# def compute_latent_beliefs_over_time(
#     df: pd.DataFrame,
#     index: tuple,
#     posterior: Posterior,
#     dt: float = 0.5,
#     padding_time: float = 0.5,
# ) -> np.ndarray[float]:
#     """
#     Compute the latent belief over time regarding the reward schedule for each box.

#     Args:
#         df: Pandas DataFrame containing session data.
#         index: Index to locate the relevant block data.
#         posterior: Instance of the Posterior class containing belief updates.
#         dt: Time bin size for discretizing the continuous-time beliefs.
#         padding_time: Additional time after the last push for modeling beliefs.

#     Returns:
#         np.ndarray: A time-based belief array of shape (n_boxes, n_timesteps, len(posterior.support())).
#     """

#     block_data = df.loc[index]
#     end_t = block_data["push times"].max() + padding_time
#     n_boxes = posterior.n_boxes
#     n_timesteps = int(end_t / dt)

#     schedule_belief_time = np.zeros((n_boxes, n_timesteps, len(posterior.support())))

#     for i in range(n_boxes):
#         box_idx = block_data["box rank"] == i
#         push_times = block_data.loc[box_idx, "push times"].to_numpy()
#         push_times = push_times[~np.isnan(push_times)]

#         # Convert posterior to discrete-time representation
#         schedule_belief_time[i] = discrete_time(
#             posterior.boxes[i].history, push_times, dt, end_t
#         )

#     return schedule_belief_time


# @process_block_safely
# def compute_reward_beliefs(
#     df: pd.DataFrame, index: tuple, posterior: Posterior
# ) -> np.ndarray:
#     """
#     Compute the belief of reward availability for each box as an event-based representation.
#     This extracts the belief about reward availability immediately before each push.

#     Args:
#         df: Pandas DataFrame containing session data.
#         index: Index to locate the relevant block data.
#         posterior: Instance of the Posterior class containing belief updates.

#     Returns:
#         np.ndarray: An event-based belief array of shape (n_obs, n_boxes, 2),
#                     where beliefs are evaluated before each push.
#     """

#     block_data = df.loc[index]

#     # Get unique schedules (sorted in descending order)
#     n_boxes = block_data["schedule"].nunique()

#     # Compute availability marginal for each push
#     n_obs = block_data["push times"].size - np.count_nonzero(
#         np.isnan(block_data["push times"])
#     )
#     belief_avail_event = np.zeros((n_obs, n_boxes, 2))
#     push_times_and_box = block_data[["push times", "box rank"]].values[:n_obs]

#     old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
#     old_t = np.zeros(n_boxes)  # Track last push time per box

#     for i, (t, box) in enumerate(push_times_and_box):
#         if t == old_t[int(box)]:
#             continue  # Skip redundant updates if push time is unchanged for the same box

#         belief_avail_event[i] = np.array(
#             [
#                 posterior.boxes[j].marginalize(
#                     obs_supp_args=[t - old_t[j]], latent_kwargs={"record": old_idx[j]}
#                 )
#                 for j in range(n_boxes)
#             ]
#         )

#         old_idx[int(box)] += 1  # Update observation index for the box
#         old_t[int(box)] = t  # Update last push time for the box

#     return belief_avail_event


def get_concurrent_reward_intervals(df: pd.DataFrame, index: tuple) -> np.ndarray:
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

    n_boxes = df["box rank"].nunique()
    block_data = df.loc[index]

    # Compute availability marginal for each push
    n_obs = block_data["push times"].size - np.count_nonzero(
        np.isnan(block_data["push times"])
    )
    belief_avail_event = np.zeros((n_obs, n_boxes))
    push_times_and_box = block_data[["reward intervals", "box rank"]].values[:n_obs]
    old_t = np.zeros(n_boxes)  # Track last push time per box

    for i, (t, box) in enumerate(push_times_and_box):
        old_t[int(box)] = t  # Update last push time for the box
        belief_avail_event[i] = old_t

    return belief_avail_event


def get_push_reward_interval_diff(df: pd.DataFrame, index: tuple) -> np.ndarray:
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

    n_boxes = df["box rank"].nunique()
    block_data = df.loc[index]

    # Compute availability marginal for each push
    n_obs = block_data["push times"].size - np.count_nonzero(
        np.isnan(block_data["push times"])
    )
    belief_avail_event = np.zeros((n_obs, n_boxes))
    push_times_and_box = block_data[
        ["reward intervals", "same-box push intervals", "box rank"]
    ].values[:n_obs]
    old_t = np.zeros(n_boxes)  # Track last push time per box

    for i, (t, pt, box) in enumerate(push_times_and_box):
        old_t[int(box)] = t  # Update last push time for the box
        belief_avail_event[i] = np.abs(old_t - pt)

    return belief_avail_event


# # todo: standardize this to return both probabilities of reward
# @process_block_safely
# def compute_reward_probabilities(
#     df: pd.DataFrame,
#     index: tuple,
#     shape: int = None,
#     schedules: list = None,
# ) -> np.ndarray:
#     """
#     Compute the exact reward probability right before each push.

#     Args:
#         df: DataFrame containing session data.
#         index: Index to locate the relevant block data.

#     Returns:
#         np.ndarray: An event-based array of shape (n_obs, n_boxes),
#                     where reward probabilities are evaluated before each push.
#     """

#     df_block = df.loc[index]
#     if schedules is None:
#         schedules = np.sort(df_block["schedule"].unique())
#     n_boxes = len(schedules)
#     if shape is None:
#         shape = df_block.index.unique("shape")[
#             0
#         ]  # Assume agent knows number of states perfectly

#     # Construct likelihood/observation model
#     obs_model = GammaObservation(shape)

#     # Compute availability marginal for each push
#     n_obs = df_block["push times"].size - np.count_nonzero(
#         np.isnan(df_block["push times"])
#     )
#     belief_avail_event = np.zeros((n_obs, n_boxes))
#     push_times_and_box = df_block[["push times", "box rank"]].values[:n_obs]
#     old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
#     old_t = np.zeros(n_boxes)  # Track last push time per box

#     for i, (t, box) in enumerate(push_times_and_box):
#         if t == old_t[int(box)]:
#             continue  # Skip redundant updates if push time is unchanged for the same box

#         belief_avail_event[i] = np.array(
#             [
#                 obs_model.probability((True, t - old_t[j]), schedules[j])
#                 for j in range(n_boxes)
#             ]
#         )

#         old_idx[int(box)] += 1  # Update observation index for the box
#         old_t[int(box)] = t  # Update last push time for the box

#     return belief_avail_event


def compute_surprise(
    df: pd.DataFrame,
    index: tuple,
    shape: int = None,
    schedules: list = None,
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
    if schedules is None:
        schedules = np.sort(df_block["schedule"].unique())
    n_boxes = len(schedules)
    if shape is None:
        shape = df_block.index.unique("shape")[
            0
        ]  # Assume agent knows number of states perfectly

    # Construct likelihood/observation model
    obs_model = GammaObservation(shape)

    # Compute availability marginal for each push
    n_obs = len(df_block)
    information = np.zeros(n_obs)
    push_ints_and_box = df_block[["same-box push intervals", "box rank"]].values[:n_obs]
    old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
    old_t = np.zeros(n_boxes)  # Track last push time per box

    for i, (t, box) in enumerate(push_ints_and_box):
        if t == old_t[int(box)]:
            continue  # Skip redundant updates if push time is unchanged for the same box
        information[i] = obs_model.surprise((True, t), schedules[int(box)])
        old_idx[int(box)] += 1  # Update observation index for the box
        old_t[int(box)] = t  # Update last push time for the box
    return information


# @process_block_safely
# def compute_fisher(
#     df: pd.DataFrame,
#     index: tuple,
#     shape: int = None,
#     schedules: list = None,
#     rate: bool = False,
# ) -> np.ndarray[float]:
#     """
#     Compute the exact reward probability of each right before each push.

#     Args:
#         df: Pandas DataFrame containing session data.
#         index: Index to locate the relevant block data.

#     Returns:
#         np.ndarray: An event-based array of shape (n_obs, n_boxes),
#                     where reward probabilities are evaluated before each push.
#     """
#     df_block = df.loc[index]
#     if schedules is None:
#         schedules = np.sort(df_block["schedule"].unique())
#     if shape is None:
#         shape = df_block.index.unique("shape")[
#             0
#         ]  # Assume agent knows number of states perfectly

#     # Construct likelihood/observation model
#     obs_model = GammaObservation(shape)

#     # Compute availability marginal for each push
#     n_obs = len(df_block)
#     information = np.zeros(n_obs)
#     push_ints_and_box = df_block[["wait times", "box rank"]].values[:n_obs]
#     for i, (t, box) in enumerate(push_ints_and_box):
#         if rate:
#             information[i] = obs_model.fisher_info_rate((True, t), schedules[int(box)])
#         else:
#             information[i] = obs_model.fisher_info((True, t), schedules[int(box)])
#     return information


# @process_block_safely
# def compute_mutual_information(
#     df: pd.DataFrame,
#     index: tuple,
#     shape: int = None,
#     schedules: list = None,
# ) -> np.ndarray[float]:
#     """
#     Compute the mutual information between the reward and the schedule.
#     """
#     df_block = df.loc[index]
#     if schedules is None:
#         schedules = np.sort(df_block["schedule"].unique())
#     n_boxes = len(schedules)
#     if shape is None:
#         shape = df_block.index.unique("shape")[
#             0
#         ]  # Assume agent knows number of states perfectly

#     def _mutual_information(t, box):
#         cond_entropy = 0

#         return np.log(6) - 0

#     # Construct likelihood/observation model
#     obs_model = GammaObservation(shape)


# @process_block_safely
# def compute_normalized_fisher(
#     df: pd.DataFrame,
#     index: tuple,
#     optimal_fisher: dict = None,
#     shape: int = None,
#     schedules: list = None,
#     rate: bool = False,
# ) -> np.ndarray[float]:
#     """
#     Compute the exact reward probability of each right before each push.

#     Args:
#         df: Pandas DataFrame containing session data.
#         index: Index to locate the relevant block data.

#     Returns:
#         np.ndarray: An event-based array of shape (n_obs, n_boxes),
#                     where reward probabilities are evaluated before each push.
#     """

#     df_block = df.loc[index]
#     if schedules is None:
#         schedules = np.sort(df_block["schedule"].unique())
#     n_boxes = len(schedules)
#     if shape is None:
#         shape = df_block.index.unique("shape")[
#             0
#         ]  # Assume agent knows number of states perfectly

#     # Construct likelihood/observation model
#     obs_model = GammaObservation(shape)

#     # Compute availability marginal for each push
#     n_obs = len(df_block)
#     information = np.zeros(n_obs)

#     push_ints_and_box = df_block[["wait times", "box rank"]].values[:n_obs]
#     old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
#     old_t = np.zeros(n_boxes)  # Track last push time per box

#     for i, (t, box) in enumerate(push_ints_and_box):
#         if t == old_t[int(box)]:
#             continue  # Skip redundant updates if push time is unchanged for the same box
#         if rate:
#             information[i] = obs_model.fisher_info_rate((True, t), schedules[int(box)])
#         else:
#             information[i] = obs_model.fisher_info((True, t), schedules[int(box)])
#         information[i] /= optimal_fisher[schedules[int(box)]]
#         old_idx[int(box)] += 1  # Update observation index for the box
#         old_t[int(box)] = t  # Update last push time for the box
#     return information


# @process_block_safely
# def compute_deriv(
#     df: pd.DataFrame,
#     index: tuple,
#     shape: int = None,
#     schedules: list = None,
# ) -> np.ndarray[float]:
#     """
#     Compute the exact reward probability of each right before each push.

#     Args:
#         df: Pandas DataFrame containing session data.
#         index: Index to locate the relevant block data.

#     Returns:
#         np.ndarray: An event-based array of shape (n_obs, n_boxes),
#                     where reward probabilities are evaluated before each push.
#     """

#     df_block = df.loc[index]
#     if schedules is None:
#         schedules = np.sort(df_block["schedule"].unique())
#     n_boxes = len(schedules)
#     if shape is None:
#         shape = df_block.index.unique("shape")[
#             0
#         ]  # Assume agent knows number of states perfectly

#     # Construct likelihood/observation model
#     obs_model = GammaObservation(shape)

#     # Compute availability marginal for each push
#     n_obs = len(df_block)
#     information = np.zeros(n_obs)
#     push_ints_and_box = df_block[["same-box push intervals", "box rank"]].values[:n_obs]
#     old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
#     old_t = np.zeros(n_boxes)  # Track last push time per box

#     for i, (t, box) in enumerate(push_ints_and_box):
#         if t == old_t[int(box)]:
#             continue  # Skip redundant updates if push time is unchanged for the same box
#         information[i] = obs_model.deriv((True, t), schedules[int(box)])
#         old_idx[int(box)] += 1  # Update observation index for the box
#         old_t[int(box)] = t  # Update last push time for the box
#     return information


# @process_block_safely
# def compute_joint_beliefs(
#     df: pd.DataFrame, index: tuple, posterior: Posterior
# ) -> np.ndarray:
#     """
#     Computes the joint belief of the reward availability and schedule of each box for each push event.

#     Args:
#         df: Pandas DataFrame containing session data with columns 'push times' and 'box rank'.
#         index: Index or identifier to select a specific block from the DataFrame.
#         posterior: An instance of the Posterior class that contains the belief states and methods
#                   for updating and computing beliefs. This posterior object is used to compute
#                   joint beliefs for the reward availability of each box.

#     Returns:
#         np.ndarray: A 4D array representing the joint beliefs for each observation, with shape
#                     (n_obs, n_boxes, 2, len(posterior.support())).
#                     The first dimension represents the observation index (push event),
#                     the second dimension represents each box,
#                     the third dimension represents the belief for reward availability (binary: 0/1),
#                     and the fourth dimension corresponds to the support of the posterior distribution.
#     """

#     n_boxes = posterior.n_boxes
#     block_data = df.loc[index]

#     # Compute availability marginal for each push
#     n_obs = block_data["push times"].size - np.count_nonzero(
#         np.isnan(block_data["push times"])
#     )
#     belief_joint_event = np.zeros((n_obs, n_boxes, 2, len(posterior.support())))
#     push_times_and_box = block_data[["push times", "box rank"]].values[:n_obs]
#     old_idx = np.zeros(n_boxes, dtype=int)  # Track last observed index per box
#     old_t = np.zeros(n_boxes)  # Track last push time per box

#     for i, (t, box) in enumerate(push_times_and_box):
#         if t == old_t[int(box)]:
#             continue  # Skip redundant updates if push time is unchanged for the same box

#         belief_joint_event[i] = np.array(
#             [
#                 posterior.boxes[j].joint(
#                     obs_supp_args=[t - old_t[j]], latent_kwargs={"record": old_idx[j]}
#                 )
#                 for j in range(n_boxes)
#             ]
#         )

#         old_idx[int(box)] += 1  # Update observation index for the box
#         old_t[int(box)] = t  # Update last push time for the box

#     return belief_joint_event


# def predict_pushed_box(
#     df: pd.DataFrame,
#     x: str | list[str],
#     y: str = "box rank",
#     perm_test: bool = False,
#     weight: bool = False,
#     n_perms: int = 500,
#     disp: bool = False,
# ) -> tuple[float, float, Any] | None:
#     """
#     Predicts the pushed box using multinomial logistic regression and evaluates the accuracy of predictions.

#     Args:
#         df: pandas DataFrame containing session data, including a column with the target labels (box rank).
#         x: A 2D numpy array containing the features (independent variables) used for prediction.
#         y: The name of the column in `df` that contains the target labels (default is 'box rank').

#     Returns:
#         A tuple containing:
#             - A float representing the accuracy of the predictions (mean of correct predictions).
#             - The fitted multinomial logistic regression model (`MNLogitResults` object).
#     """
#     X = df[x]
#     y = df[y]
#     try:
#         # mdl = smf.mnlogit("y ~ X", {'y': y, 'X': X}).fit(disp = disp)
#         # yhat = np.argmax(mdl.predict(), axis=1)
#         # accuracy = (yhat == y).mean()
#         weights = (
#             compute_sample_weight(class_weight="balanced", y=y) if weight else None
#         )
#         mdl = LogisticRegression()
#         mdl.fit(X, y, sample_weight=weights)
#         accuracy = mdl.score(X, y, sample_weight=weights)
#         rsq = mcfadden_pseudo_rsquared(mdl, X, y)
#         if perm_test:
#             pval_accu, pval_rsq = permutation_test_logistic(
#                 X, y, accuracy, rsq, weights=weights, n_perms=n_perms
#             )
#             return accuracy, rsq, pval_accu, pval_rsq, mdl
#         return accuracy, rsq, mdl
#     except Exception as e:
#         logger.debug(e)
#         return None


def get_mean_beliefs_over_time(
    beliefs: ArrayLike | list, supp: ArrayLike | list
) -> ArrayLike | list:
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


def get_map_over_time(
    beliefs: ArrayLike | list, supp: ArrayLike | list
) -> ArrayLike | list:
    """
    Computes the maximum a posteriori (MAP) beliefs over time.

    Args:
        beliefs: Matrix or list of beliefs for each box over time.
        supp: The support for the beliefs (values over which the beliefs are computed).

    Returns:
        A vector of MAP estimates over time.
    """
    supp = np.asarray(supp)
    if isinstance(beliefs, np.ndarray):
        # Find the index of maximum probability along the support dimension
        map_indices = np.argmax(beliefs, axis=-1)
        # Return the corresponding support values
        return supp[map_indices]
    return [supp[np.argmax(x)] for x in beliefs]


def get_map_indices_over_time(
    beliefs: ArrayLike | list, supp: ArrayLike | list
) -> ArrayLike | list:
    """
    Computes the indices of the maximum a posteriori (MAP) beliefs over time.

    Args:
        beliefs: Matrix or list of beliefs for each box over time.
        supp: The support for the beliefs (values over which the beliefs are computed).

    Returns:
        A vector of MAP indices over time.
    """
    supp = np.asarray(supp)
    if isinstance(beliefs, np.ndarray):
        # Find the index of maximum probability along the support dimension
        map_indices = np.argmax(beliefs, axis=-1)
        # Return the corresponding support values
        return map_indices
    return [np.argmax(x) for x in beliefs]


def get_std_beliefs_over_time(
    beliefs: ArrayLike | list, supp: ArrayLike | list
) -> np.ndarray | list:
    """
    Computes the std of beliefs over time.

    Args:
        beliefs: Matrix or list of beliefs for each box over time.
        supp: Support of the beliefs.

    Returns:
        A vector of the std of beliefs over time.
    """
    if isinstance(beliefs, np.ndarray):
        return np.sqrt(
            beliefs @ (supp[:, np.newaxis] ** 2) - (beliefs @ supp[:, np.newaxis]) ** 2
        ).squeeze()
    return [np.sqrt(x @ (supp**2) - (x @ supp) ** 2) for x in beliefs]


def get_entropy_beliefs_over_time(beliefs: ArrayLike | list) -> np.ndarray | list:
    if isinstance(beliefs, np.ndarray):
        return entropy(beliefs, axis=1)
    return [entropy(x) for x in beliefs]
