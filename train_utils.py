import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils import data

EPSILON = 1e-9

def load_signal_data(dataset_dir: str) -> tuple:
    """Load the x_train, x_test, y_train and y_test from the data dir as torch tensors.

    Parameters
    ----------
    dataset_dir: str
        Where the processed data (scaled and all) is stored.

    Returns
    -------
    tuple
        tuple with train and test data as torch Tensors.
    """

    test_labels: torch.Tensor = torch.from_numpy(
        pd.read_csv(os.path.join(dataset_dir, 'y_test.csv'), squeeze=True).values.astype(
            'float32'))

    train_labels: torch.Tensor = torch.from_numpy(
        pd.read_csv(
            os.path.join(dataset_dir, 'y_train.csv'), squeeze=True).values.astype(
            'float32')
        )

    train_data: torch.Tensor = torch.from_numpy(
        pd.read_csv(os.path.join(dataset_dir, 'X_train_processed.csv')).values
    ).float()

    test_data: torch.Tensor = torch.from_numpy(
        pd.read_csv(os.path.join(dataset_dir, 'X_test_processed.csv')).values
    ).float()

    return train_data, train_labels, test_data, test_labels


def load_datasets(data_dir: str) -> tuple:
    """Load the train and test PyTorch dataset for given dataset parameter.

    Parameters
    ----------
    data_dir: str
        Where the data should be saved/loaded from.

    Returns
    -------
    tuple
        Tuple with PyTorch train and test dataset.
    """

    train_data, train_labels, test_data, test_labels = load_signal_data(data_dir)
    train_data = data.TensorDataset(train_data, train_labels)
    test_data = data.TensorDataset(test_data, test_labels)

    return train_data, test_data


def cross_entropy(y_true: np.array, y_score: np.array, n_labels: int = None,
                  reduction: str = None) -> np.array:
    """Calculate cross entropy between y_true and y_score

    Parameters
    ----------
    y_true: np.array
        Labels
    y_score: np.array
        Predictions
    n_labels: int (optional)
        Only needs to be set in multiclass case.
    reduction: str
        If and how the cross entropies should be reduced. Options: ['sum', 'mean']

    Returns
    -------
    np.array
        Cross entropy losses of the predictions
    """
    # Squeeze the array if possible.
    try:
        y_score = y_score.squeeze(axis=-1)
    except ValueError:
        pass

    # If the array has 2 dimensions treat as single-class.
    try:
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]
    except IndexError:
        pass

    # Calculate binary cross entropy or multiclass cross entropy.
    if len(y_score.shape) == 1:

        if y_true.shape != y_score.shape:
            raise ValueError("y_true shape not equal to y_score shape")

        ce = -(y_true * np.log2(y_score + EPSILON) + (1 - y_true) * np.log2(1 - y_score + EPSILON))

    else:
        # One-hot encode y_true
        y_true = np.eye(n_labels)[y_true]

        if y_true.shape != y_score.shape:
            raise ValueError("y_true shape not equal to y_score shape")

        ce = -np.sum(y_true * np.log2(y_score + EPSILON), axis=-1)

    # Apply reduction.
    if reduction == 'sum':
        return ce.sum()
    if reduction == 'mean':
        return ce.mean()

    return ce


def make_performance_uncertainty_plot(y_true: np.array,
                                      y_pred: np.array,
                                      y_unc: np.array,
                                      y_axis_label: str,
                                      performance_fn: callable = cross_entropy,
                                      performance_fn_args: dict = None) -> plt.figure:
    """Create plot how the uncertainty relates to model performance.

    Parameters
    ----------
    y_true: np.array
        True labels
    y_pred: np.array
        Predictions
    y_unc: np.array
        Uncertainties
    y_axis_label: str
        plot Y-axis label
    performance_fn: callable
        Performance function used
    performance_fn_args: dict
        Arguments passed to performance function

    Returns
    -------
    plt.figure
        Plot
    """
    try:
        y_unc.squeeze(-1)

    except ValueError:
        pass

    if y_unc.ndim == 2:
        y_unc = y_unc.mean(-1)

    elif y_unc.ndim > 2:
        raise ValueError(f"Invalid uncertainty shape: {y_unc.shape}")

    if y_true.ndim != 1:
        raise ValueError("Y-true not one-dimensional")

    # Placeholder
    if performance_fn_args is None:
        performance_fn_args = {}

    order = y_unc.argsort()

    sorted_uncertainties = y_unc[order]
    sorted_labels = y_true[order]
    sorted_predictions = y_pred[order]

    # Get the first index where both 0's and 1's have occurred with at least a batch size of 64.
    first_index = max(128,
                      np.argwhere(sorted_labels != sorted_labels[0])[0][0])
    performances = []
    percentages = []

    for i in range(first_index + 1, len(sorted_uncertainties)):
        selected_labels = sorted_labels[:i]
        selected_predictions = sorted_predictions[:i]

        percentages.append(100 * len(selected_predictions) / len(y_pred))

        performances.append(performance_fn(selected_labels, selected_predictions,
                                           **performance_fn_args))

    fig = plt.figure()
    sns.lineplot(percentages, performances)
    plt.xlabel('% of Uncertain Data')
    plt.ylabel(y_axis_label)
    return fig

