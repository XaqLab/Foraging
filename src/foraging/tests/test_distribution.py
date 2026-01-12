"""
Unit tests for the distribution module.

This module tests Bayesian inference, posterior updates, and all distribution-related classes.
"""
import numpy as np
import pytest
from scipy.stats import gamma

from foraging.models.distribution import (
    Probabilities,
    ExactBayesianUpdateOnProbabilities,
    RewardOutcomeLikelihood,
    PermutationLikelihood,
    Posterior,
    FactorizedBelief,
    FactorizedPosterior,
    RewardOutcome,
    IndexedObservation,
    GammaParameters,
    Permutation,
    PossibleSchedules
)


def test_probabilities_initialization():
    """Test proper initialization of probabilities."""
    support = [1, 2, 3, 4]
    probabilities = [0.1, 0.2, 0.3, 0.4]
    prob_dist = Probabilities(support, probabilities)
    
    np.testing.assert_array_equal(prob_dist.support, support)
    np.testing.assert_array_equal(prob_dist.representation, probabilities)


def test_probability_validation():
    """Test that probabilities are properly validated."""
    # Test non-negative constraint
    with pytest.raises(AssertionError):
        Probabilities([1, 2], [-0.1, 1.1])

    # Test normalization constraint
    with pytest.raises(AssertionError):
        Probabilities([1, 2], [0.3, 0.3])

    # Test valid probabilities
    valid_probs = Probabilities([1, 2], [0.3, 0.7])
    np.testing.assert_array_equal(valid_probs.representation, [0.3, 0.7])


def test_probabilities_query():
    """Test querying probabilities for specific values."""
    prob_dist = Probabilities([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    
    assert prob_dist.query(1) == 0.1
    assert prob_dist.query(2) == 0.2
    assert prob_dist.query(3) == 0.3
    assert prob_dist.query(4) == 0.4


def test_probabilities_sampling():
    """Test sampling from the distribution."""
    prob_dist = Probabilities([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    np.random.seed(42)
    samples = list(prob_dist.sample(1000))
    
    # Count samples
    sample_counts = {val: samples.count(val) for val in [1, 2, 3, 4]}
    total_samples = sum(sample_counts.values())
    
    # Check that sampling approximately follows the distribution
    for val, expected_prob in zip([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]):
        observed_prob = sample_counts[val] / total_samples
        assert abs(observed_prob - expected_prob) < 0.1


def test_bayesian_update_manual_calculation():
    """Test Bayesian update with manual calculation verification."""
    support = [1, 2, 3, 4]
    prior_probs = [0.25, 0.25, 0.25, 0.25]
    prior = Probabilities(support, prior_probs)
    
    def simple_likelihood(observation, parameter):
        if observation == "high":
            return [0.1, 0.2, 0.3, 0.4][parameter - 1]
        else:
            return 0.5
    
    bayesian_update = ExactBayesianUpdateOnProbabilities(simple_likelihood)
    observation = "high"
    
    # Manual calculation of posterior
    likelihoods = [0.1, 0.2, 0.3, 0.4]
    unnormalized_posterior = [l * p for l, p in zip(likelihoods, prior_probs)]
    normalization_constant = sum(unnormalized_posterior)
    expected_posterior = [p / normalization_constant for p in unnormalized_posterior]
    
    # Apply Bayesian update
    posterior = bayesian_update(prior, observation)
    
    # Verify posterior probabilities
    np.testing.assert_array_equal(posterior.representation, expected_posterior)
    assert np.sum(posterior.representation) == 1.0


def test_bayesian_update_sequential_updates():
    """Test multiple sequential Bayesian updates with manual verification."""
    support = [1, 2, 3, 4]
    prior_probs = [0.25, 0.25, 0.25, 0.25]
    prior = Probabilities(support, prior_probs)
    
    def simple_likelihood(observation, parameter):
        if observation == "high":
            return [0.1, 0.2, 0.3, 0.4][parameter - 1]
        elif observation == "low":
            return [0.4, 0.3, 0.2, 0.1][parameter - 1]
        else:
            return 0.5
    
    bayesian_update = ExactBayesianUpdateOnProbabilities(simple_likelihood)
    
    # First update
    posterior1 = bayesian_update(prior, "high")
    
    # Second update using posterior1 as prior
    posterior2 = bayesian_update(posterior1, "low")
    
    # Manual calculation of expected posterior after two sequential updates
    # Step 1: First update with "high" observation
    likelihoods_high = [0.1, 0.2, 0.3, 0.4]
    unnormalized_posterior1 = [l * p for l, p in zip(likelihoods_high, prior_probs)]
    normalization1 = sum(unnormalized_posterior1)
    expected_posterior1 = [p / normalization1 for p in unnormalized_posterior1]
    
    # Step 2: Second update with "low" observation using posterior1 as prior
    likelihoods_low = [0.4, 0.3, 0.2, 0.1]
    unnormalized_posterior2 = [l * p for l, p in zip(likelihoods_low, expected_posterior1)]
    normalization2 = sum(unnormalized_posterior2)
    expected_posterior2 = [p / normalization2 for p in unnormalized_posterior2]
    
    # Verify that posterior2 matches manually calculated values
    np.testing.assert_array_almost_equal(
        posterior2.representation, expected_posterior2, decimal=10
    )
    assert np.sum(posterior2.representation) == 1.0


def test_bayesian_update_vectorized():
    """Test vectorized Bayesian update."""
    support = [1, 2, 3, 4]
    prior_probs = [0.25, 0.25, 0.25, 0.25]
    prior = Probabilities(support, prior_probs)
    
    def vectorized_likelihood(observation, support):
        if observation == "high":
            return np.array([0.1, 0.2, 0.3, 0.4])
        else:
            return np.array([0.5, 0.5, 0.5, 0.5])
    
    vectorized_update = ExactBayesianUpdateOnProbabilities(vectorized_likelihood, vectorize=True)
    posterior = vectorized_update(prior, "high")
    
    # Manual calculation of expected posterior for vectorized update
    likelihoods = np.array([0.1, 0.2, 0.3, 0.4])
    unnormalized_posterior = likelihoods * np.array(prior_probs)
    expected_posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    
    # Verify that posterior matches manually calculated values
    np.testing.assert_array_almost_equal(
        posterior.representation, expected_posterior, decimal=10
    )
    assert np.sum(posterior.representation) == 1.0


def test_gamma_likelihood_available_reward():
    """Test likelihood calculation for available reward."""
    likelihood = RewardOutcomeLikelihood()
    gamma_params = GammaParameters(shape=2.0, schedule=5.0)
    obs = RewardOutcome(is_available=True, time=3.0)
    
    # Calculate expected probability manually
    expected_prob = gamma.cdf(3.0, 2.0, scale=5.0/2.0)
    
    likelihood_value = likelihood(obs, gamma_params)
    assert abs(likelihood_value - expected_prob) < 1e-10


def test_gamma_likelihood_unavailable_reward():
    """Test likelihood calculation for unavailable reward."""
    likelihood = RewardOutcomeLikelihood()
    gamma_params = GammaParameters(shape=2.0, schedule=5.0)
    obs = RewardOutcome(is_available=False, time=3.0)
    
    # Calculate expected probability manually
    expected_prob = 1.0 - gamma.cdf(3.0, 2.0, scale=5.0/2.0)
    
    likelihood_value = likelihood(obs, gamma_params)
    assert abs(likelihood_value - expected_prob) < 1e-10


def test_gamma_likelihood_edge_cases():
    """Test likelihood at edge cases."""
    likelihood = RewardOutcomeLikelihood()
    gamma_params = GammaParameters(shape=2.0, schedule=5.0)
    
    # Time = 0
    obs_zero = RewardOutcome(is_available=True, time=0.0)
    likelihood_zero = likelihood(obs_zero, gamma_params)
    assert likelihood_zero >= 0.0
    assert likelihood_zero <= 1.0
    
    # Very large time
    obs_large = RewardOutcome(is_available=True, time=100.0)
    likelihood_large = likelihood(obs_large, gamma_params)
    assert abs(likelihood_large - 1.0) < 1e-5


def test_permutation_likelihood_available_reward():
    """Test likelihood calculation for available reward in permutation."""
    likelihood = PermutationLikelihood()
    permutation = Permutation(permutation=[5.0, 3.0, 7.0], shape=2.0)
    
    obs = IndexedObservation(
        i=1, 
        observation=RewardOutcome(is_available=True, time=2.0)
    )
    
    # Calculate expected probability manually for index 1 (schedule=3.0)
    expected_prob = gamma.cdf(2.0, 2.0, scale=3.0/2.0)
    
    likelihood_value = likelihood(obs, permutation)
    assert abs(likelihood_value - expected_prob) < 1e-10


def test_permutation_likelihood_unavailable_reward():
    """Test likelihood calculation for unavailable reward in permutation."""
    likelihood = PermutationLikelihood()
    permutation = Permutation(permutation=[5.0, 3.0, 7.0], shape=2.0)
    
    obs = IndexedObservation(
        i=0, 
        observation=RewardOutcome(is_available=False, time=1.0)
    )
    
    # Calculate expected probability manually for index 0 (schedule=5.0)
    expected_prob = 1.0 - gamma.cdf(1.0, 2.0, scale=5.0/2.0)
    
    likelihood_value = likelihood(obs, permutation)
    assert abs(likelihood_value - expected_prob) < 1e-10


def test_posterior_initialization():
    """Test posterior initialization."""
    support = [1, 2, 3, 4]
    prior_probs = [0.25, 0.25, 0.25, 0.25]
    prior = Probabilities(support, prior_probs)
    
    def simple_likelihood(observation, parameter):
        if observation == "high":
            return [0.1, 0.2, 0.3, 0.4][parameter - 1]
        else:
            return 0.5
    
    update_rule = ExactBayesianUpdateOnProbabilities(simple_likelihood)
    posterior = Posterior(init_id="init", prior=prior, update=update_rule)
    
    assert posterior.prior == prior
    assert len(posterior) == 1
    assert "init" in posterior


def test_posterior_update():
    """Test posterior belief updates."""
    support = [1, 2, 3, 4]
    prior_probs = [0.25, 0.25, 0.25, 0.25]
    prior = Probabilities(support, prior_probs)
    
    def simple_likelihood(observation, parameter):
        if observation == "high":
            return [0.1, 0.2, 0.3, 0.4][parameter - 1]
        else:
            return 0.5
    
    update_rule = ExactBayesianUpdateOnProbabilities(simple_likelihood)
    posterior = Posterior(init_id="init", prior=prior, update=update_rule)
    
    # Update with observation
    posterior.update(key="obs1", o="high")
    
    # Check that posterior was updated
    assert len(posterior) == 2
    assert "obs1" in posterior
    
    # Check that head is updated
    head_belief = posterior.head
    assert abs(np.sum(head_belief.representation) - 1.0) < 1e-10


def test_posterior_query():
    """Test querying the current belief."""
    support = [1, 2, 3, 4]
    prior_probs = [0.25, 0.25, 0.25, 0.25]
    prior = Probabilities(support, prior_probs)
    
    def simple_likelihood(observation, parameter):
        if observation == "high":
            return [0.1, 0.2, 0.3, 0.4][parameter - 1]
        else:
            return 0.5
    
    update_rule = ExactBayesianUpdateOnProbabilities(simple_likelihood)
    posterior = Posterior(init_id="init", prior=prior, update=update_rule)
    
    # Query before update
    initial_query = posterior.query(1)
    assert initial_query == 0.25
    
    # Update and query again
    posterior.update(key="obs1", o="high")
    updated_query = posterior.query(1)
    
    # Should be different due to Bayesian update
    assert initial_query != updated_query


def test_factorized_belief_initialization():
    """Test factorized belief initialization."""
    support = [1, 2, 3]
    probabilities = [0.33, 0.33, 0.34]
    base_prior = Probabilities(support, probabilities)
    n_factors = 3
    factorized_belief = FactorizedBelief(n_factors, base_prior)
    
    assert len(factorized_belief) == n_factors
    
    # Check that each factor is a copy of the base prior
    for i in range(n_factors):
        factor = factorized_belief[i]
        np.testing.assert_array_almost_equal(factor.representation, probabilities)


def test_factorized_belief_query():
    """Test querying joint probability."""
    support = [1, 2, 3, 4]
    probabilities = [0.25, 0.25, 0.25, 0.25]
    base_prior = Probabilities(support, probabilities)
    factorized_belief = FactorizedBelief(3, base_prior)
    
    # Query joint probability of [1, 1, 1]
    joint_prob = factorized_belief.query([1, 1, 1])
    expected_prob = 0.25 * 0.25 * 0.25  # Product of individual probabilities
    assert abs(joint_prob - expected_prob) < 1e-6


def test_factorized_posterior_update():
    """Test updating factorized posterior with indexed observation."""
    support = [1, 2, 3, 4]
    probabilities = [0.25, 0.25, 0.25, 0.25]  # Uniform probabilities over 4 integers
    base_prior = Probabilities(support, probabilities)
    
    def simple_likelihood(observation, parameter):
        if observation == "high":
            return [0.1, 0.2, 0.3, 0.4][parameter - 1]
        else:
            return 0.5
    
    update_rule = ExactBayesianUpdateOnProbabilities(simple_likelihood)
    factorized_posterior = FactorizedPosterior(
        n_factors=2,
        init_id="init",
        update=update_rule,
        prior=base_prior
    )
    
    # Create indexed observation for factor 0
    indexed_obs = IndexedObservation(i=0, observation="high")
    
    # Update posterior
    factorized_posterior.update(key="obs1", o=indexed_obs)
    
    # Check that posterior was updated
    assert len(factorized_posterior) == 2
    
    # Check that only factor 0 was updated
    head = factorized_posterior.head
    factor_0_prob = head[0].query(1)
    factor_1_prob = head[1].query(1)
    
    # Factor 0 should be updated (different from prior)
    # Factor 1 should remain the same
    assert abs(factor_0_prob - 0.25) > 1e-10
    assert abs(factor_1_prob - 0.25) < 0.01


def test_reward_observation():
    """Test RewardObservation data structure."""
    obs = RewardOutcome(is_available=True, time=2.5)
    assert obs.is_available
    assert obs.time == 2.5


def test_indexed_observation():
    """Test IndexedObservation data structure."""
    reward_obs = RewardOutcome(is_available=False, time=1.0)
    indexed_obs = IndexedObservation(i=2, observation=reward_obs)
    
    assert indexed_obs.i == 2
    assert indexed_obs.observation == reward_obs


def test_gamma_parameters():
    """Test GammaParameters data structure."""
    params = GammaParameters(shape=2.0, schedule=5.0)
    assert params.shape == 2.0
    assert params.schedule == 5.0


def test_permutation():
    """Test Permutation data structure."""
    permutation = Permutation(permutation=[1.0, 2.0, 3.0], shape=2.0)
    assert permutation.permutation == [1.0, 2.0, 3.0]
    assert permutation.shape == 2.0


def test_possible_schedules():
    """Test PossibleSchedules data structure."""
    schedules = [1.0, 2.0, 3.0, 4.0]
    possible_schedules = PossibleSchedules(shape=2.0, schedule=schedules)
    
    assert possible_schedules.shape == 2.0
    assert possible_schedules.schedule == schedules
    
    # Test array conversion
    array = np.asarray(possible_schedules)
    np.testing.assert_array_equal(array, schedules)
    
    # Test iteration
    assert list(possible_schedules) == schedules
    
    # Test indexing
    assert possible_schedules[0] == 1.0
    assert possible_schedules[3] == 4.0
    
    # Test length
    assert len(possible_schedules) == 4


def test_complete_bayesian_inference_pipeline():
    """Test complete Bayesian inference with gamma likelihood."""
    support = np.array([1, 2, 3, 4])
    prior_probs = [0.25, 0.25, 0.25, 0.25]
    prior = Probabilities(support=PossibleSchedules(schedule=support, shape=10), probabilities=prior_probs)
    
    likelihood = RewardOutcomeLikelihood()
    update_rule = ExactBayesianUpdateOnProbabilities(likelihood, vectorize=True)
    posterior = Posterior(init_id="init", prior=prior, update=update_rule)
    
    # Create observations
    observations = [
        RewardOutcome(is_available=True, time=1.0),
        RewardOutcome(is_available=False, time=2.0),
        RewardOutcome(is_available=True, time=0.5)
    ]
    
    # Update posterior with observations
    for i, obs in enumerate(observations):
        posterior.update(key=f"obs_{i}", o=obs)
    
    # Check that posterior evolved
    assert len(posterior) == 4  # init + 3 observations
    
    # Check normalization
    final_belief = posterior.head
    assert abs(np.sum(final_belief.representation) - 1.0) < 1e-10
    
    # Check that probabilities changed from uniform prior
    assert not np.allclose(final_belief.representation, prior_probs)


def test_factorized_bayesian_inference_pipeline():
    """Test complete factorized Bayesian inference pipeline."""
    support = np.array([1, 2, 3, 4])
    prior_probs = [0.25, 0.25, 0.25, 0.25]
    prior = Probabilities(support=PossibleSchedules(schedule=support, shape=10), probabilities=prior_probs)
    
    likelihood = RewardOutcomeLikelihood()
    update_rule = ExactBayesianUpdateOnProbabilities(likelihood, vectorize=True)
    
    # Set up factorized posterior
    factorized_posterior = FactorizedPosterior(
        n_factors=2,
        init_id="init",
        update=update_rule,
        prior=prior
    )
    
    # Create indexed observations
    indexed_observations = [
        IndexedObservation(i=0, observation=RewardOutcome(is_available=True, time=1.0)),
        IndexedObservation(i=1, observation=RewardOutcome(is_available=False, time=2.0)),
        IndexedObservation(i=0, observation=RewardOutcome(is_available=True, time=0.5))
    ]
    
    # Update factorized posterior
    for i, obs in enumerate(indexed_observations):
        factorized_posterior.update(key=f"obs_{i}", o=obs)
    
    # Check that posterior evolved
    assert len(factorized_posterior) == 4
    
    # Check that factors are properly updated
    final_belief = factorized_posterior.head
    assert len(final_belief) == 2
    
    # Both factors should be normalized
    for i in range(2):
        factor_probs = final_belief[i].representation
        assert abs(np.sum(factor_probs) - 1.0) < 1e-10