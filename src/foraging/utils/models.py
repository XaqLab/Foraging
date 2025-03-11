import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import product
from typing import Optional, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gamma


class Distribution(ABC):
    """
    An abstract class to represent any distribution
    """

    @abstractmethod
    def probability(self, *args, **kwargs) -> Any:
        """
        Calculates the probability of a given value.

        Returns:
            The likelihood of the observation given the latent variable.
        """

        pass

    @abstractmethod
    def probabilities(self, *args, **kwargs) -> Any:
        """
        Returns the probabilities of all members in the support of the distribution

        Returns:
            The probabilities over the support
        """
        pass

    @abstractmethod
    def support(self, *args, **kwargs) -> Any:
        """
        Returns the support of the distribution. The support is the set of values
        for which the probability of the observation is non-zero or valid.

        Returns:
            The support of the observation distribution.
        """

        pass

class Observation(Distribution):
    """
    An abstract class to represent an observation in an experiment. This class defines the
    structure for calculating the probability (likelihood) of an observation given a latent variable.

    Methods
    -------
    probability(obs, latent, *args, **kwargs):
        Calculates the probability (likelihood) of the given observation given a latent variable.

    probabilities(latent, *args, **kwargs):
        Returns the probability of all possible observations under a given latent variable.
    """

    @abstractmethod
    def probability(self, obs: Any, latent: Any, *args, **kwargs) -> Any:
        """
        Calculates the probability (likelihood) of the observation given the latent variable.

        Args:
            obs: The observed data (can vary based on the implementation).
            latent: The latent variable that potentially generated the observation.

        Returns:
            The likelihood of the observation given the latent variable.
        """
        pass

    @abstractmethod
    def probabilities(self, latent: Any, *args, **kwargs) -> Iterable:
        """
        Returns the probabilities of all possible observations under a latent variable

        Args:
            latent: The latent variable that potentially generated the observations.

        Returns:
            The probabilities over the support
        """
        pass

class Normalizer(ABC):
    """
    A class to abstractly represent the normalization of a distribution.

    This class defines the structure for normalizing a probability distribution. It ensures that the
    probability distribution is properly scaled so that the sum of the probabilities equals 1.

    Methods
    -------
    normalize(probs, *args, **kwargs):
        Normalize the provided distribution of probabilities.
    """

    @abstractmethod
    def normalize(self, probs: ArrayLike, *args, **kwargs) -> ArrayLike:
        """
        Normalize the provided distribution of probabilities.

        Args:
            probs (array-like): The distribution to normalize, typically an array of probabilities.
            **kwargs: Additional arguments specific to the normalization process.

        Returns:
            The normalized probability distribution, where the sum of probabilities equals 1.
        """
        pass

class Posterior(Distribution):
    """
    An abstract class to represent a posterior (belief). This class defines the structure
    for representing a posterior distribution and the methods for calculating the probability
    of a latent variable, updating the posterior according to new observations, and retrieving
    the support of the posterior distribution.

    Methods
    -------
    prior(*args, **kwargs):
        Returns the prior that implements the abstract class Distribution

    observation_model(*args, **kwargs):
        Returns the observation model that implements the abstract class Observation.

    normalizer(*args, **kwargs):
        Returns an object used to normalize the posterior distribution.

    probability(latent, *args, **kwargs):
        Returns the probability of the latent variable according to the current posterior.

    update(obs, *args, **kwargs):
        Updates the posterior according to the observation using Bayes' rule.
    """

    @abstractmethod
    def prior(self, *args, **kwargs) -> Distribution:
        """
        Returns prior

        Returns:
            Distribution: An object that implements the Distribution class
        """
        pass

    @abstractmethod
    def observation_model(self, *args, **kwargs) -> Observation:
        """
        Returns the observation model that implements the abstract class Observation.

        Returns:
            Observation: An object that implements the Observation class.
        """
        pass

    @abstractmethod
    def normalizer(self, *args, **kwargs) -> Normalizer:
        """
        Returns an object that normalizes the posterior distribution.

        Returns:
            object: The normalizer used for posterior distribution.
        """
        pass

    @abstractmethod
    def probability(self, latent: Any, *args, **kwargs) -> float:
        """
        Returns the probability of the latent variable according to the current posterior.

        Args:
            latent: The latent variable whose probability is to be calculated.

        Returns:
            float: The probability of the latent variable according to the posterior.
        """
        pass

    @abstractmethod
    def update(self, obs: Any, *args, **kwargs):
        """
        Updates the posterior according to the observation using Bayes' rule.

        Args:
            obs: The observation to update the posterior with.

        Returns:
            None
        """
        pass

class GammaObservation(Observation):
    """
    An implementation of P(observations | reward appearance rate, number of states), the likelihood model in the extended Poisson process.

    This class models the likelihood of observing a certain event based on the reward appearance rate and the number of states in the system. The model is based on a Gamma distribution, which is used to calculate the likelihood of reward availability.

    Methods
    -------
    probability(obs: tuple[bool, float], latent: float, *args, **kwargs):
        Computes the probability of the observation given the reward availability rate.
    probabilities(latent: Any, t: float | Iterable, *args, **kwargs):
        Computes the probability of all possible observations given the reward availability rate and possible push interval(s).
    support(t: float | Iterable):
        Returns the support of the observation distribution, which in this case is the cartesian product of reward outcomes and push interval(s) t e.g. [(0, t1), (0, t2)...(1,tn)].
    """

    def __init__(self, shape: int):
        """
        Initializes the GammaObservation object.

        Args:
            shape (int): The shape parameter of the Gamma distribution, which defines the number of states in the model.
        """
        self.shape = shape

    def probability(self, obs: tuple[bool, float], latent: float | ArrayLike, *args, **kwargs) -> float | ArrayLike:
        """
        Computes the probability of the observation given the reward appearance rate and the number of states.

        Args:
            obs (tuple[bool, float]): A tuple consisting of a boolean indicating whether the reward was observed (1 if yes, 0 if no) and the wait time before pushing the box.
            latent (float | ArrayLike): The latent mean schedule(s) whose likelihood we would like to assess.

        Returns:
            The probability/probabilities of observing the event given the latent reward availability rate(s).
        """
        # Extract the reward availability and push interval.
        is_avail = obs[0]
        t = obs[1]

        # Calculate the probability of reward being available/unavailable after t time has passed under the given latents.
        p_t = gamma.cdf(t, self.shape, scale= latent / self.shape)

        # The last element is the probability of being in the last state, i.e., reward being available.
        if not is_avail:  # If reward is not available, return the complementary probability.
            return 1.0 - p_t
        return p_t

    def probabilities(self, latent: Any, t: float | Iterable, *args, **kwargs) -> ArrayLike:
        supp = self.support(t)
        probs = []
        for obs in supp:
            probs.append(self.probability(obs, latent))
        return probs

    def support(self, t: float | Iterable, *args, **kwargs) -> product:
        """
        Returns the support of the observation distribution as a function of possible push intervals t.

        Args:
            t: array of push intervals
        Returns:
            numpy.ndarray: The possible observations.
        """
        t = np.atleast_1d(t)
        return itertools.product([False, True], t)

class MyPrior(Distribution):
    def __init__(self, probs: ArrayLike, support: ArrayLike):
        self.probs = probs
        self.supp = support
    def probability(self, index: int, *args, **kwargs) -> Any:
        return self.probs[index]
    def probabilities(self, *args, **kwargs) -> Any:
        return self.probs
    def support(self, *args, **kwargs) -> Any:
        return self.supp

class MyPosterior(Posterior):
    """
    A generic implementation of a posterior that performs Bayesian updates on latent variable.

    This class implements the `Posterior` abstract class, providing functionality
    for Bayesian updates based on an observation model, prior, and support.

    Methods
    -------
    prior():
        Returns the prior probabilities used for initialization.

    observation_model():
        Returns the observation model used for calculating likelihoods.

    normalizer():
        Returns the normalizer used to normalize probabilities.

    probability(index):
        Retrieves the probability of the latent variable at the given index.

    probabilities():
        Retrieves the probabilities of all latent variables.

    update(obs):
        Updates the posterior according to Bayes' rule given an observation.

    reset():
        Resets the posterior to the prior.

    support():
        Returns the support of the posterior distribution.
    """

    def __init__(self, obs_model: Observation, prior: ArrayLike, support: ArrayLike,
                 normalizer: Optional[Normalizer] = None, record: bool = False):
        """
        Constructs a posterior from the given inputs.

        Args:
            obs_model (Observation): The observation model used to calculate the likelihood.
            prior: An array containing the prior probabilities that the posterior is initialized to.
            support: An array enumerating the support of the posterior.
            normalizer (Normalizer): The normalizer used to normalize the probabilities (optional).
            record (bool): flag indicating whether to maintain a record of previous beliefs
        """
        self.obs_model = obs_model
        if normalizer is None:
            normalizer = MyNormalizer()
        prior = normalizer.normalize(prior)
        self._prior = MyPrior(prior, support)
        self.norm = normalizer
        self.probs = prior
        self.supp = support
        self.record = record
        if record:
            self.history = [self.probs]

    def prior(self) -> Distribution:
        """
        Returns prior
        Returns:
            prior
        """
        return self._prior

    def observation_model(self) -> Observation:
        """
        Returns the observation model used for calculating likelihoods.

        Returns:
            The observation model.
        """
        return self.obs_model

    def normalizer(self) -> Normalizer:
        """
        Returns the normalizer used to normalize probabilities.

        Returns:
            The normalizer object.
        """
        return self.norm

    def probability(self, index: int, *args, record: Optional[int] = -1, **kwargs) -> float:
        """
        Retrieves the probability of the latent variable at the given index.

        Args:
            index (int): The index into the array of posterior probabilities to retrieve.
            record (int): if the history of beliefs is maintained, specify this parameter to index into history
        Returns:
            The probability of the latent variable taking on the indexed value.
        """
        if self.record and record > -1:
            return self.history[record][index]

        return self.probs[index]

    def probabilities(self, *args, record: Optional[int | str] = -1, **kwargs) -> ArrayLike:
        if self.record and record == 'all':
            return self.history
        elif self.record and record > -1:
            return self.history[record]
        return self.probs

    def update(self, obs: Any, **kwargs) -> ArrayLike:
        """
        Updates the posterior according to Bayes' rule given an observation.

        Args:
            obs: The observation to update the posterior.
            **kwargs: Additional arguments.
            - 'obs_args': positional arguments (passed to `obs_model.probability`).
            - 'obs_kwargs': keyword arguments (passed to `obs_model.probability`).
            - 'latent_kwargs': keyword arguments (passed to `self.probabilities`).
            - 'norm_args': positional arguments (passed to `normalizer.normalize`).
            - 'norm_kwargs': keyword arguments (passed to `normalizer.normalize`).

        Returns:
            An array containing the updated posterior probabilities.
        """
        obs_model = self.observation_model()
        normalizer = self.normalizer()
        support = self.support()
        posterior_probs = obs_model.probability(obs, support, *kwargs.pop('obs_args', tuple()), **kwargs.pop('obs_kwargs', {})) * self.probabilities(**kwargs.pop('latent_kwargs', {}))

        # Normalize the new posterior
        self.probs = normalizer.normalize(posterior_probs, *kwargs.pop('norm_args', tuple()), **kwargs.pop('norm_kwargs',
                                                                                                         {}))
        # If keep history, then add current posterior to buffer
        if self.record:
            self.history.append(self.probs)
        return self.probs

    def joint(self, **kwargs) -> ArrayLike:
        """
        Computes the joint distribution of latent and possible observations.

        Args:
            **kwargs: Additional arguments.
            - 'obs_supp_args': positional arguments (passed to `obs_model.support`).
            - 'obs_supp_kwargs': keyword arguments (passed to `obs_model.support`).
            - 'latent_supp_args': positional arguments (passed to `self.support`).
            - 'latent_supp_kwargs': keyword arguments (passed to `self.support`).
            - 'obs_args': positional arguments (passed to `obs_model.probability`).
            - 'obs_kwargs': keyword arguments (passed to `obs_model.probability`).
            - 'latent_kwargs': keyword arguments (passed to `self.probabilities`).
            - 'norm_args': positional arguments (passed to `normalizer.normalize`).
            - 'norm_kwargs': keyword arguments (passed to `normalizer.normalize`).

        Returns:
            np.ndarray: A matrix of joint probabilities, where each row represents an observation, and each column represents a possible value of the latent variable.
        """
        # Retrieve the observation model
        obs_model = self.observation_model()

        # Get the support of the observation model, with optional arguments from kwargs
        obs_support = list(obs_model.support(*kwargs.pop('obs_supp_args', tuple()), **kwargs.pop('obs_supp_kwargs', {})))
        # Get the support of the latent distribution (the possible latent values)
        latent_support = self.support(*kwargs.pop('latent_supp_args', tuple()), **kwargs.pop('latent_supp_kwargs', {}))

        # Initialize an empty joint probability matrix
        joint = np.zeros((len(obs_support), len(latent_support)))

        # Iterate through all possible combinations of observations and latent values
        for i, o in enumerate(obs_support):
            p1 = obs_model.probability(o, latent_support, *kwargs.pop('obs_args', tuple()), **kwargs.pop('obs_kwargs', {}))
            p2 = self.probabilities(**kwargs.pop('latent_kwargs', {}))
            joint[i] = p1 * p2
        return joint

    def marginalize(self, **kwargs) -> ArrayLike:
        """
        Marginalizes over latent to get distribution of possible observations
        Args:
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: array of observation probabilities
        """
        return self.joint(**kwargs).sum(axis = 1).squeeze()

    def reset(self, empty_history: bool = False):
        """
        Resets the posterior to the prior.

        This method is used to reinitialize the posterior with the prior probabilities.
        """
        self.probs = self.prior()
        if self.record:
            if empty_history:
                self.history = [self.probs]
            else:
                self.history.append(self.probs)

    def support(self, *args, **kwargs):
        """
        Returns the support of the posterior distribution.

        Returns:
            The support array of the posterior distribution.
        """
        return self.supp

class IndependentBoxesPosterior(Posterior):

    def __init__(self, n_boxes: int, obs_model: Observation, prior: ArrayLike, support: ArrayLike,
                 normalizer: Optional[Normalizer] = None, **kwargs):
        self.obs_model = obs_model
        if normalizer is None:
            normalizer = MyNormalizer()
        self.norm = normalizer
        self.supp = support
        self.boxes = [MyPosterior(obs_model, prior, support, normalizer, **kwargs) for _ in range(n_boxes)]
        self.n_boxes = n_boxes

    def prior(self, *args, **kwargs) -> Distribution:
        """
        Return priors from all boxes

        Args:
            *args:
            **kwargs:

        Returns:
            list of priors
        """
        return self.boxes[0].prior()

    def observation_model(self, *args, **kwargs) -> Observation:
        """
        Return observation model from all boxes

        Args:
            *args:
            **kwargs:

        Returns:
            list of observation models
        """
        return self.obs_model

    def normalizer(self, *args, **kwargs) -> Normalizer:
        """
        Return normalizer

        Args:
            *args:
            **kwargs:

        Returns:
            normalizer
        """
        return self.norm

    def probabilities(self, *args, **kwargs) -> list[ArrayLike]:
        """
        Return posterior probabilities from all boxes
        Args:
            *args:
            **kwargs:

        Returns:
            list of posterior probabilities
        """
        return [box.probabilities(*args, **kwargs) for box in self.boxes]


    def probability(self, index: int, *args, **kwargs) -> list[float]:
        """
        Retrieves the probability of the latent variable at the given index for all boxes.

        Args:
            index (int): The index into the array of posterior probabilities to retrieve.
            *args:
            **kwargs:

        Returns:
            The probability of the latent variable taking on the indexed value.
        """
        return [box.probability(index, *args, **kwargs) for box in self.boxes]

    def update(self, obs: Any, box: int, *args, **kwargs) -> ArrayLike:
        """
        Updates the posterior at the specified box according to Bayes' rule given an observation.

        Args:
            obs: The observation to update the posterior.
            box (int): box to retrieve posterior from.
            *args:
            **kwargs:

        Returns:
            An array containing the updated posterior probabilities.
        """
        return self.boxes[box].update(obs, **kwargs)

    def joint(self, **kwargs) -> list[ArrayLike]:
        """
        Computes the joint distribution of latent and possible observations.

        Args:
            **kwargs:
        Returns:
            np.ndarray: A matrix of joint probabilities, where each row represents an observation, and each column represents a possible value of the latent variable.
        """
        return [box.joint(**kwargs) for box in self.boxes]

    def marginalize(self, **kwargs) -> list[ArrayLike]:
        """
        Marginalizes over latent to get distribution of possible observations
        Args:
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: array of observation probabilities
        """
        return [box.joint(**kwargs).sum(axis = 1).squeeze() for box in self.boxes]

    def support(self, *args, **kwargs) -> ArrayLike:
        """
        Returns the support of the posterior distribution.

        Returns:
            The support array of the posterior distribution.
        """
        return self.supp

#todo: this will be correctly implemented later
class DependentBoxesPosterior(MyPosterior):
    def __init__(self, obs_model: Observation, prior: ArrayLike, support: ArrayLike,
                 normalizer: Optional[Normalizer] = None):
        super().__init__(obs_model, prior, support, normalizer)

    def probability(self, index: int, box: int = 0, **kwargs) -> float:
        """
        Retrieves the probability of the latent variable at the given index.

        Args:
            index: The index into the array of posterior probabilities to retrieve.
            box: box to retrieve posterior from.
            **kwargs: Additional arguments.

        Returns:
            The probability of the latent variable taking on the indexed value.
        """
        return self.probs[box, index]

    def update(self, obs: Any, box: int = 0, **kwargs) -> ArrayLike:
        """
        Updates the posterior according to Bayes' rule given an observation.

        Args:
            obs: The observation to update the posterior.
            box: box to retrieve posterior from.
            **kwargs: Additional arguments.

        Returns:
            An array containing the updated posterior probabilities.
        """
        obs_model = self.observation_model()
        normalizer = self.normalizer()
        support = self.support()
        posterior_probs = np.zeros(len(support))

        # Implements Pr(x|o_{1:t}) proportional to L(x|o_t) * Pr(x|o_{1:t-1})
        for i, x in enumerate(support):
            posterior_probs[box, i] = obs_model.probability(obs, x, *kwargs.pop('obs_args', tuple()), **kwargs.pop('obs_kwargs',
                                                                                                         {})) * self.probability(i, box)
        # todo: update all boxes to reflect knowledge of ordering --> p(schedule & this box is fastest) or p(schedule & this box is middle) or p(schedule & this box is slowest)
        # p(this box is fastest| experience) = p(this box's schedule > other boxes')

        # Normalize the new posterior
        self.probs = normalizer.normalize(posterior_probs[box], *kwargs.pop('norm_args', tuple()), **kwargs.pop('norm_kwargs',
                                                                                                         {}))
        return self.probs

class MyNormalizer(Normalizer):
    """
    A generic implementation of a normalization scheme for a distribution.

    Methods
    -------
    normalize(probs, **kwargs):
        Normalizes the provided distribution `probs` by dividing by the sum of its probabilities.
    """

    def normalize(self, probs: ArrayLike, **kwargs) -> ArrayLike:
        """
        Normalizes distribution `probs` by the sum of its probabilities.

        Args:
            probs: The distribution to normalize.
            **kwargs: Additional parameters (if any).

        Returns:
            The array of normalized probabilities.
        """
        denom = np.sum(probs)
        if denom == 0:
            return np.zeros_like(probs)
        return probs / denom
