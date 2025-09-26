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

# %%
import numpy as np
from hexarena.color import get_cmap, get_cue_movie, get_cue_movie_independent_noise
from numpy.fft import fftfreq, ifft2

from foraging.constants import SEED
from foraging.plotting.video import display_frames_video, frames_to_rgb
from foraging.psychophysics import (
    ActualExperimentColorCue,
    IndependentNoiseColorCue,
    PsychophysicsExperiment,
)
from foraging.utils.autoreload import setup_hexarena

setup_hexarena()

RNG = np.random.default_rng(SEED)
SIZE = (16, 16)

# %% [markdown]
# # Example Stimulus
#
# ## Actual noise structure

# %%
kappa = 0.1
cue = 0.0

frames = get_cue_movie(cue, 10, size=SIZE, kappa=kappa)
frames_rgb = frames_to_rgb(frames, get_cmap())

# Display as video
display_frames_video(frames_rgb, fps=1, duration=10.0)

# %% [markdown]
# ## Independent noise structure

# %%
frames = get_cue_movie_independent_noise(cue, 10, size=SIZE, kappa=kappa)
frames_rgb = frames_to_rgb(frames, get_cmap())

# Display as video
display_frames_video(frames_rgb, fps=1, duration=10.0)

# %%


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


# %%
