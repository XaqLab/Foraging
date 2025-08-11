"""
Psychophysics Experiment for Cue Discrimination with Time-Series Data

This module implements a psychophysics experiment where an agent must discriminate
between two different cue values based on time-series of images they generate.
The experiment compares performance under independent vs correlated noise conditions.
"""

import logging
from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from hexarena.color import get_cue_movie, get_cue_movie_independent_noise
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from foraging.config.constants import PSYCHOPHYSICS_IMAGE_SIZE, SEED
from foraging.utils import kwargs_handler
from foraging.utils.autoreload import setup_hexarena

setup_hexarena()

logger = logging.getLogger(__name__)


class StimulusGenerator(ABC):
    """
    Generates image stimuli encoding cue values with custom noise.
    Outputs 3D arrays where first dimension is time and remaining dimensions are image dimensions.
    """

    @abstractmethod
    def __init__(self, image_size: tuple[int, int] = PSYCHOPHYSICS_IMAGE_SIZE):
        """
        Initialize the time-series image generator.

        Args:
            image_size: Size of the generated images (image_size x image_size)
        """
        self.image_size = image_size

    @abstractmethod
    def generate_stimuli(
        self, cue_value: float, noise_param: Any, time_steps: int = 10
    ) -> np.ndarray:
        """
        Generate a time-series of images from a cue value with noise.

        Args:
            cue_value: The underlying cue value
            time_steps: Number of time steps in the sequence

        Returns:
            Generated stimuli as numpy array of shape (time_steps, height, width)
        """
        pass


class ActualExperimentColorCue(StimulusGenerator):
    """
    Generates color cue stimuli for the actual experiment.
    """

    def __init__(self, image_size: tuple[int, int] = PSYCHOPHYSICS_IMAGE_SIZE):
        super().__init__(image_size)

    def generate_stimuli(
        self,
        cue_value: float,
        time_steps: int = 10,
        noise_param: float = 0.01,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate a time-series of images from a cue value with noise.
        """
        return get_cue_movie(
            cue_value, time_steps, size=self.image_size, kappa=noise_param, **kwargs
        )


class IndependentNoiseColorCue(StimulusGenerator):
    """
    Generates color cue stimuli with independent noise.
    """

    def __init__(self, image_size: tuple[int, int] = PSYCHOPHYSICS_IMAGE_SIZE):
        super().__init__(image_size)

    def generate_stimuli(
        self,
        cue_value: float,
        time_steps: int = 10,
        noise_param: float = 0.01,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate a time-series of images from a cue value with noise.
        """
        return get_cue_movie_independent_noise(
            cue_value, time_steps, size=self.image_size, kappa=noise_param, **kwargs
        )


class PsychophysicsDataset(Dataset):
    """
    Dataset for the psychophysics experiment.
    """

    def __init__(
        self,
        n_samples: int,
        stimulus_generator: StimulusGenerator,
        cue_values: List[float] = [0.3, 0.7],
        image_size: tuple[int, int] = PSYCHOPHYSICS_IMAGE_SIZE,
        time_steps: int = 10,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            n_samples: Number of samples per cue value
            stimulus_generator: Stimulus generator
            cue_values: List of two cue values to discriminate between
            image_size: Size of the generated images
            time_steps: Number of time steps in the sequence
            **kwargs: Additional arguments for the stimulus generator
        """
        self.cue_values = cue_values
        self.image_size = image_size
        self.time_steps = time_steps
        self.stimulus_generator = stimulus_generator
        self.stimulus, self.labels = self._generate_data(n_samples, **kwargs)

    def _generate_data(
        self, n_samples: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the dataset.

        Args:
            n_samples: Number of samples per cue value
            **kwargs: Additional arguments for the stimulus generator
        """
        time_series = []
        labels = []

        for i, cue_value in enumerate(self.cue_values):
            for _ in range(n_samples):
                ts = self.stimulus_generator.generate_stimuli(
                    cue_value, self.time_steps, **kwargs
                )
                time_series.append(ts)
                labels.append(i)

        return np.array(time_series), np.array(labels)

    def __len__(self):
        return len(self.stimulus)

    def __getitem__(self, idx):
        # Shape: (time_steps, height, width) -> (time_steps, channels, height, width)
        stimulus = torch.FloatTensor(self.stimulus[idx]).unsqueeze(
            1
        )  # Add channel dimension
        label = torch.LongTensor([self.labels[idx]])
        return stimulus, label


# TODO: consider whether agent trained to do sequential categorization is better than one trained to see both images simultaneously as a single concatenated image
class Agent(nn.Module):
    """
    RNN with CNN frontend for cue discrimination.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = PSYCHOPHYSICS_IMAGE_SIZE,
        n_classes: int = 2,
        cnn_channels: List[int] = [4, 8],
        rnn_hidden_size: int = 16,
        rnn_layers: int = 2,
        dropout: float = 0.2,
        fix_features: bool = False,
    ):
        """
        Initialize the agent.

        Args:
            image_size: Size of input images
            n_classes: Number of classes to discriminate
            cnn_channels: List of channel sizes for CNN layers
            rnn_hidden_size: Hidden size of RNN
            rnn_layers: Number of RNN layers
            dropout: Dropout rate
            fix_features: Whether to freeze the features of the CNN
        """
        super().__init__()

        self.image_size = image_size
        self.n_classes = n_classes

        # CNN frontend for spatial feature extraction
        cnn_layers = []
        in_channels = 1

        for out_channels in cnn_channels:
            cnn_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)
        if fix_features:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # Calculate CNN output size dynamically
        with torch.no_grad():
            # Create a dummy input to compute the actual output size
            height, width = self.image_size
            dummy_input = torch.randn(1, 1, height, width)
            dummy_output = self.cnn(dummy_input)
            cnn_output_size = dummy_output.view(-1).size(0)

        # RNN layers for temporal processing
        self.rnn = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_size // 2, n_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time-series data.

        Args:
            x: Input tensor of shape (batch_size, time_steps, channels, height, width)

        Returns:
            Output logits
        """
        batch_size, time_steps, channels, height, width = x.size()

        # Process each time step through CNN
        cnn_features = []
        for t in range(time_steps):
            # Extract features for current time step
            features = self.cnn(
                x[:, t, :, :, :]
            )  # (batch_size, cnn_channels[-1], h, w)
            features = features.view(batch_size, -1)  # Flatten
            cnn_features.append(features)

        # Stack features across time steps
        cnn_features = torch.stack(
            cnn_features, dim=1
        )  # (batch_size, time_steps, cnn_output_size)

        # RNN processing
        rnn_out, _ = self.rnn(cnn_features)  # (batch_size, time_steps, hidden_size)

        # Take the last output
        rnn_out = rnn_out[:, -1, :]  # (batch_size, hidden_size)

        # Classification
        logits = self.classifier(rnn_out)  # (batch_size, n_classes)

        return logits


class PsychophysicsExperiment:
    """
    Main experiment class for cue discrimination psychophysics.
    """

    def __init__(
        self,
        true_stimulus_generator: StimulusGenerator,
        surrogate_stimulus_generator: StimulusGenerator,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: tuple[int, int] = PSYCHOPHYSICS_IMAGE_SIZE,
        time_steps: int = 10,
    ):
        """
        Initialize the experiment.

        Args:
            device: Device to run the experiment on
            image_size: Size of the generated images
            time_steps: Number of time steps in sequences
            true_stimulus_generator: Generator for true stimuli
            surrogate_stimulus_generator: Generator for surrogate stimuli
        """
        self.device = device
        self.image_size = image_size
        self.time_steps = time_steps
        self.results = {}
        self.true_stimulus_generator = true_stimulus_generator
        self.surrogate_stimulus_generator = surrogate_stimulus_generator

        logger.info(f"Initializing psychophysics experiment on device: {device}")

    def run_experiment(
        self,
        true_noise_params: List[float] = [0.01, 0.1],
        surrogate_noise_params: List[float] = [0.01, 0.1],
        n_samples_per_condition: int = 1000,
        cue_values: List[float] = [0.3, 0.7],
        train_split: float = 0.8,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        n_epochs: int = 50,
        early_stopping_patience: int = 10,
    ) -> Dict:
        """
        Run the complete psychophysics experiment.

        Args:
            true_noise_params: List of true noise parameters to test
            surrogate_noise_params: List of surrogate noise parameters to test
            n_samples_per_condition: Number of samples per condition
            cue_values: Cue values to discriminate between
            train_split: Fraction of data for training
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            n_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary with experiment results
        """
        results = {"true_noise": {}, "surrogate_noise": {}, "comparison": {}}

        # Test independent noise conditions
        logger.info("Testing true stimulus conditions...")
        for true_noise_param in tqdm(
            true_noise_params, desc="True stimulus experiment"
        ):
            performance = self.perceptual_training(
                stimulus_generator=self.true_stimulus_generator,
                n_samples=n_samples_per_condition,
                cue_values=cue_values,
                noise_param=true_noise_param,
                train_split=train_split,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
            )
            results["true_noise"][true_noise_param] = performance

        # Test correlated noise conditions
        logger.info("Testing surrogate noise conditions...")
        for surrogate_noise_param in tqdm(
            surrogate_noise_params, desc="Surrogate noise experiment"
        ):
            performance = self.perceptual_training(
                stimulus_generator=self.surrogate_stimulus_generator,
                noise_param=surrogate_noise_param,
                n_samples=n_samples_per_condition,
                cue_values=cue_values,
                train_split=train_split,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
            )
            results["surrogate_noise"][surrogate_noise_param] = performance

        # Find equivalent performance
        equivalent_parameters = self._find_equivalent_performance(
            results["true_noise"], results["surrogate_noise"]
        )

        results["comparison"]["equivalent_parameters"] = equivalent_parameters

        self.results = results
        return results

    def perceptual_training(
        self,
        stimulus_generator: StimulusGenerator,
        noise_param: Any,
        n_samples: int,
        cue_values: List[float],
        train_split: float,
        batch_size: int,
        learning_rate: float,
        n_epochs: int,
        early_stopping_patience: int,
        **kwargs,
    ) -> Dict:
        """Test a specific noise condition."""

        # Create datasets
        dataset_kwargs = kwargs_handler(kwargs, "dataset_kwargs")
        dataset = PsychophysicsDataset(
            n_samples=n_samples,
            stimulus_generator=stimulus_generator,
            cue_values=cue_values,
            noise_param=noise_param,
            image_size=self.image_size,
            time_steps=self.time_steps,
            **dataset_kwargs,
        )

        # Split into train/val with equal label fractions
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=1 - train_split,  # Or desired split ratio
            stratify=dataset.labels,
            random_state=SEED,  # For reproducibility
        )

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        agent_kwargs = kwargs_handler(kwargs, "agent_kwargs")
        model = Agent(
            image_size=self.image_size, n_classes=len(cue_values), **agent_kwargs
        ).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.squeeze().to(
                    self.device
                )

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.squeeze().to(
                        self.device
                    )

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    predictions = torch.argmax(outputs, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            val_acc = accuracy_score(val_targets, val_predictions)
            val_accuracies.append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        final_predictions = []
        final_targets = []
        final_probabilities = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.squeeze().to(
                    self.device
                )

                outputs = model(batch_x)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                final_predictions.extend(predictions.cpu().numpy())
                final_targets.extend(batch_y.cpu().numpy())
                final_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        final_acc = accuracy_score(final_targets, final_predictions)
        final_auc = roc_auc_score(final_targets, [p[1] for p in final_probabilities])

        # Calculate d-prime (sensitivity index)
        hit_rate = np.mean(
            [p == t for p, t in zip(final_predictions, final_targets) if t == 1]
        )
        false_alarm_rate = np.mean(
            [p == 1 for p, t in zip(final_predictions, final_targets) if t == 0]
        )

        # Adjust for perfect performance
        hit_rate = min(0.99, max(0.01, hit_rate))
        false_alarm_rate = min(0.99, max(0.01, false_alarm_rate))

        d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)

        return {
            "accuracy": final_acc,
            "auc": final_auc,
            "d_prime": d_prime,
            "hit_rate": hit_rate,
            "false_alarm_rate": false_alarm_rate,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "predictions": final_predictions,
            "targets": final_targets,
            "probabilities": final_probabilities,
        }

    def _find_equivalent_performance(
        self, true_results: Dict, surrogate_results: Dict
    ) -> Optional[float]:
        """Find the parameter where true and surrogate performance are equivalent."""

        # Use d-prime as the performance metric
        true_dprimes = {k: v["d_prime"] for k, v in true_results.items()}
        surrogate_dprimes = {k: v["d_prime"] for k, v in surrogate_results.items()}

        # Find the parameter where performances are closest
        min_diff = float("inf")
        equivalent_param = None

        for param in true_dprimes.keys():
            if param in surrogate_dprimes:
                diff = abs(true_dprimes[param] - surrogate_dprimes[param])
                if diff < min_diff:
                    min_diff = diff
                    equivalent_param = param

        return equivalent_param

    def plot_results(self, save_path: Optional[str] = None):
        """Plot the experiment results."""
        if not self.results:
            logger.warning("No results to plot. Run experiment first.")
            return

        # Extract parameter values
        true_noise_params = sorted(list(self.results["true_noise"].keys()))
        surrogate_noise_params = sorted(list(self.results["surrogate_noise"].keys()))

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: True noise performance
        true_dprimes = [
            self.results["true_noise"][param]["d_prime"] for param in true_noise_params
        ]
        true_accuracies = [
            self.results["true_noise"][param]["accuracy"] for param in true_noise_params
        ]

        axes[0, 0].plot(
            true_noise_params,
            true_dprimes,
            "o-",
            label="d-prime",
            linewidth=2,
            markersize=8,
        )
        axes[0, 0].set_xlabel("True Noise Parameter")
        axes[0, 0].set_ylabel("d-prime")
        axes[0, 0].set_title("True Stimulus Performance")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Add accuracy on secondary y-axis
        ax2 = axes[0, 0].twinx()
        ax2.plot(
            true_noise_params,
            true_accuracies,
            "s--",
            color="red",
            label="accuracy",
            alpha=0.7,
        )
        ax2.set_ylabel("Accuracy", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Plot 2: Surrogate noise performance
        surrogate_dprimes = [
            self.results["surrogate_noise"][param]["d_prime"]
            for param in surrogate_noise_params
        ]
        surrogate_accuracies = [
            self.results["surrogate_noise"][param]["accuracy"]
            for param in surrogate_noise_params
        ]

        axes[0, 1].plot(
            surrogate_noise_params,
            surrogate_dprimes,
            "o-",
            label="d-prime",
            linewidth=2,
            markersize=8,
        )
        axes[0, 1].set_xlabel("Surrogate Noise Parameter")
        axes[0, 1].set_ylabel("d-prime")
        axes[0, 1].set_title("Surrogate Stimulus Performance")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Add accuracy on secondary y-axis
        ax2 = axes[0, 1].twinx()
        ax2.plot(
            surrogate_noise_params,
            surrogate_accuracies,
            "s--",
            color="red",
            label="accuracy",
            alpha=0.7,
        )
        ax2.set_ylabel("Accuracy", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Plot 3: Performance comparison
        # Find common parameter range for comparison
        all_params = sorted(list(set(true_noise_params + surrogate_noise_params)))

        # Interpolate to common parameter values if needed
        true_interp = []
        surrogate_interp = []
        common_params = []

        for param in all_params:
            if param in true_noise_params and param in surrogate_noise_params:
                true_interp.append(self.results["true_noise"][param]["d_prime"])
                surrogate_interp.append(
                    self.results["surrogate_noise"][param]["d_prime"]
                )
                common_params.append(param)

        if common_params:
            axes[1, 0].plot(
                common_params,
                true_interp,
                "o-",
                label="True Stimulus",
                linewidth=2,
                markersize=8,
            )
            axes[1, 0].plot(
                common_params,
                surrogate_interp,
                "s--",
                label="Surrogate Stimulus",
                linewidth=2,
                markersize=8,
            )
            axes[1, 0].set_xlabel("Noise Parameter")
            axes[1, 0].set_ylabel("d-prime")
            axes[1, 0].set_title("Performance Comparison")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # Mark equivalent performance point
            if self.results["comparison"]["equivalent_parameters"]:
                eq_param = self.results["comparison"]["equivalent_parameters"]
                if eq_param in common_params:
                    idx = common_params.index(eq_param)
                    axes[1, 0].plot(
                        eq_param,
                        true_interp[idx],
                        "ro",
                        markersize=12,
                        label=f"Equivalent: {eq_param:.3f}",
                    )
                    axes[1, 0].legend()

        # Plot 4: Performance difference
        if common_params:
            performance_diff = np.array(true_interp) - np.array(surrogate_interp)
            axes[1, 1].plot(
                common_params,
                performance_diff,
                "o-",
                color="purple",
                linewidth=2,
                markersize=8,
            )
            axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
            axes[1, 1].set_xlabel("Noise Parameter")
            axes[1, 1].set_ylabel("d-prime difference (True - Surrogate)")
            axes[1, 1].set_title("Performance Difference")
            axes[1, 1].grid(True)

            # Mark zero crossing (equivalent performance)
            if self.results["comparison"]["equivalent_parameters"]:
                eq_param = self.results["comparison"]["equivalent_parameters"]
                if eq_param in common_params:
                    idx = common_params.index(eq_param)
                    axes[1, 1].plot(
                        eq_param,
                        performance_diff[idx],
                        "ro",
                        markersize=12,
                        label=f"Equivalent: {eq_param:.3f}",
                    )
                    axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def generate_sample_time_series(self, noise_scale: float = 1.0, n_samples: int = 4):
        """Generate and display sample time-series from both noise conditions."""
        fig, axes = plt.subplots(
            2, n_samples, self.time_steps, figsize=(3 * n_samples, 2 * self.time_steps)
        )

        # Independent noise samples
        ind_generator = IndependentNoiseColorCue(self.image_size)
        for i in range(n_samples):
            cue_value = 0.3 if i < n_samples // 2 else 0.7
            time_series = ind_generator.generate_stimuli(
                cue_value, self.time_steps, noise_param=noise_scale
            )
            for t in range(self.time_steps):
                axes[0, i, t].imshow(time_series[t], cmap="gray")
                if t == 0:
                    axes[0, i, t].set_title(f"Independent\nCue={cue_value:.1f}")
                axes[0, i, t].axis("off")

        # Correlated noise samples
        cor_generator = ActualExperimentColorCue(self.image_size)
        for i in range(n_samples):
            cue_value = 0.3 if i < n_samples // 2 else 0.7
            time_series = cor_generator.generate_stimuli(
                cue_value, self.time_steps, noise_param=noise_scale
            )
            for t in range(self.time_steps):
                axes[1, i, t].imshow(time_series[t], cmap="gray")
                if t == 0:
                    axes[1, i, t].set_title(f"Correlated\nCue={cue_value:.1f}")
                axes[1, i, t].axis("off")

        plt.suptitle(f"Sample Time-Series (Noise Scale = {noise_scale})")
        plt.tight_layout()
        plt.show()


def run_demo_experiment():
    """Run a demonstration of the psychophysics experiment."""
    print("Running Psychophysics Experiment Demo...")

    # Initialize experiment
    experiment = PsychophysicsExperiment(
        time_steps=10,
        true_stimulus_generator=ActualExperimentColorCue(),
        surrogate_stimulus_generator=IndependentNoiseColorCue(),
    )

    # Run experiment with fewer samples for demo
    print("Running experiment...")
    results = experiment.run_experiment(
        true_noise_params=[0.005, 0.01, 0.02],
        surrogate_noise_params=[0.005, 0.01, 0.02],
        n_samples_per_condition=500,
        n_epochs=20,
        early_stopping_patience=5,
    )

    # Plot results
    print("Plotting results...")
    experiment.plot_results()

    # Print summary
    print("\nExperiment Summary:")
    print(f"Equivalent noise scale: {results['comparison']['equivalent_parameters']}")

    for noise_type in ["independent_noise", "correlated_noise"]:
        print(f"\n{noise_type.replace('_', ' ').title()}:")
        for scale, perf in results[noise_type].items():
            print(
                f"  Scale {scale}: Accuracy={perf['accuracy']:.3f}, d-prime={perf['d_prime']:.3f}"
            )

    return results


if __name__ == "__main__":
    run_demo_experiment()
