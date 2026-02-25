from typing import Iterable

import torch
import torch.nn as nn


class PolicyMLP(nn.Module):
    """
    Multi-task MLP for predicting push intervals and box choices from belief vectors.

    Architecture:
    - Shared hidden layers with ReLU activation and dropout
    - Separate heads for regression (push interval) and classification (box choice)
    """

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], n_boxes: int):
        """
        Initialize the PolicyMLP.

        Args:
            input_dim: Dimension of input belief vectors
            hidden_dims: List of hidden layer dimensions
            n_boxes: Number of possible box choices (output classes)
        """
        super(PolicyMLP, self).__init__()

        # Shared layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)

        # Separate heads
        self.interval_head = nn.Linear(prev_dim, 1)  # Regression for push interval
        self.box_head = nn.Linear(prev_dim, n_boxes)  # Classification for box choice

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input belief vectors of shape (batch_size, input_dim)

        Returns:
            interval_pred: Predicted push intervals of shape (batch_size, 1)
            box_logits: Box choice logits of shape (batch_size, n_boxes)
        """
        shared_out = self.shared(x)
        interval_pred = self.interval_head(shared_out)
        box_logits = self.box_head(shared_out)
        return interval_pred, box_logits

class ActionMLP(nn.Module):
    """MLP for predicting discretized actions from belief vectors."""
    
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], n_actions: int):
        super(ActionMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        self.shared = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, n_actions)
    
    def forward(self, x):
        shared_out = self.shared(x)
        action_logits = self.action_head(shared_out)
        return action_logits
