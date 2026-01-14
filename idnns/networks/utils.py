"""Utility functions for data loading and processing."""
import numpy as np
import scipy.io as sio
import os
import sys


def load_data(name, random_labels=False):
    """Load the data.

    Args:
        name: The name of the dataset (e.g., 'MNIST' or path to .mat file)
        random_labels: True if we want to return random labels to the dataset

    Returns:
        Object with data and labels attributes
    """
    print('Loading Data...')
    C = type('type_C', (object,), {})
    data_sets = C()

    if name.split('/')[-1] == 'MNIST':
        from torchvision import datasets, transforms

        # Download/load MNIST
        data_dir = os.path.join(os.path.dirname(sys.argv[0]), "data")
        os.makedirs(data_dir, exist_ok=True)

        train_data = datasets.MNIST(root=data_dir, train=True, download=True)
        test_data = datasets.MNIST(root=data_dir, train=False, download=True)

        # Convert to numpy and normalize to [0, 1]
        train_images = train_data.data.numpy().reshape(-1, 784) / 255.0
        test_images = test_data.data.numpy().reshape(-1, 784) / 255.0
        data_sets.data = np.concatenate([train_images, test_images], axis=0).astype(np.float32)

        # One-hot encode labels
        train_labels = np.eye(10)[train_data.targets.numpy()]
        test_labels = np.eye(10)[test_data.targets.numpy()]
        data_sets.labels = np.concatenate([train_labels, test_labels], axis=0).astype(np.float32)
    else:
        # Load from .mat file
        d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
        F = d['F']
        y = d['y']
        data_sets.data = F.astype(np.float32)
        data_sets.labels = np.squeeze(
            np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T
        ).astype(np.float32)

    # If we want to assign random labels to the data
    if random_labels:
        labels = np.zeros(data_sets.labels.shape, dtype=np.float32)
        labels_index = np.random.randint(low=0, high=labels.shape[1], size=labels.shape[0])
        labels[np.arange(len(labels)), labels_index] = 1
        data_sets.labels = labels

    return data_sets


def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly."""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
    """Divide the data to train and test and shuffle it.

    Args:
        data_sets_org: Original dataset with data and labels
        percent_of_train: List containing percentage for training data
        min_test_data: Minimum percentage of data to use for testing
        shuffle_data: Whether to shuffle the data

    Returns:
        Object with train and test subsets
    """
    def perc(i, t):
        # Handle numpy arrays by extracting scalar value
        val = np.asarray(i).flatten()[0] if hasattr(i, '__iter__') else i
        return int(np.rint((val * t) / 100))

    C = type('type_C', (object,), {})
    data_sets = C()

    # Extract scalar from percent_of_train[0] if it's an array
    train_pct = np.asarray(percent_of_train[0]).flatten()[0]

    stop_train_index = perc(train_pct, data_sets_org.data.shape[0])
    start_test_index = stop_train_index

    if train_pct > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.data.shape[0])

    data_sets.train = C()
    data_sets.test = C()

    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(
            data_sets_org.data, data_sets_org.labels
        )
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels

    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:, :]

    return data_sets
