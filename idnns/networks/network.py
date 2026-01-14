"""Train and calculate the information of network - PyTorch version."""
import multiprocessing
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from idnns.information import information_process as inn
from idnns.networks.utils import data_shuffle
from idnns.networks import model as mo

warnings.filterwarnings("ignore")
NUM_CORES = multiprocessing.cpu_count()

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(activation_function, layerSize, input_size, num_of_classes,
                learning_rate_local, save_file, covn_net):
    """Build specific model of the network.

    Returns the network model.
    """
    model = mo.Model(input_size, layerSize, num_of_classes, learning_rate_local,
                     save_file, int(activation_function), cov_net=covn_net)
    return model


def train_and_calc_inf_network(i, j, k, layerSize, num_of_ephocs, learning_rate_local,
                               batch_size, indexes, save_grads, data_sets_org,
                               model_type, percent_of_train, interval_accuracy_display,
                               calc_information, calc_information_last, num_of_bins,
                               interval_information_display, save_ws, rand_int, cov_net):
    """Train the network and calculate its information."""
    network_name = '{0}_{1}_{2}_{3}'.format(i, j, k, rand_int)
    print('Training network  - {0}'.format(network_name))
    network = train_network(layerSize, num_of_ephocs, learning_rate_local, batch_size,
                            indexes, save_grads, data_sets_org, model_type,
                            percent_of_train, interval_accuracy_display, network_name,
                            cov_net)
    network['information'] = []
    if calc_information:
        print('Calculating the information')
        information = np.array([inn.get_information(network['ws'], data_sets_org.data,
                                                    data_sets_org.labels, num_of_bins,
                                                    interval_information_display,
                                                    network['model'], layerSize)])
        network['information'] = information
    elif calc_information_last:
        print('Calculating the information for the last epoch')
        information = np.array([inn.get_information([network['ws'][-1]], data_sets_org.data,
                                                    data_sets_org.labels, num_of_bins,
                                                    interval_information_display,
                                                    network['model'], layerSize)])
        network['information'] = information
    # If we don't want to save layer's output
    if not save_ws:
        network['weights'] = 0
    return network


def extract_activity(model, batch_points_all, data_sets_org, device=DEVICE):
    """Get the activation values of the layers for the input.

    PyTorch version using forward pass with activation capture.
    """
    model.train(False)
    w_temp = None

    with torch.no_grad():
        for i in range(0, len(batch_points_all) - 1):
            batch_xs = data_sets_org.data[batch_points_all[i]:batch_points_all[i + 1]]
            batch_tensor = torch.FloatTensor(batch_xs).to(device)

            # Forward pass populates model.hidden_layers
            _ = model(batch_tensor)

            # Get hidden layer activations
            hidden_activations = [h.cpu().numpy() if isinstance(h, torch.Tensor) else h
                                  for h in model.hidden_layers]

            if w_temp is None:
                w_temp = hidden_activations
            else:
                for s in range(len(hidden_activations)):
                    w_temp[s] = np.concatenate((w_temp[s], hidden_activations[s]), axis=0)

    model.train(True)
    return w_temp


def print_accuracy(model, data_sets, batch_size, epoch, acc_train_array, device=DEVICE):
    """Calculate the test accuracy and print train and test accuracy."""
    model.train(False)
    correct = 0
    total = 0

    with torch.no_grad():
        test_data = torch.FloatTensor(data_sets.test.data).to(device)
        test_labels = torch.FloatTensor(data_sets.test.labels).to(device)

        for i in range(0, len(test_data), batch_size):
            batch_x = test_data[i:i + batch_size]
            batch_y = test_labels[i:i + batch_size]

            outputs = model(batch_x)
            pred = outputs.argmax(dim=1)
            target = batch_y.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += batch_y.size(0)

    test_acc = correct / total if total > 0 else 0
    train_acc = np.mean(np.array(acc_train_array)) if acc_train_array else 0

    print('Epoch {0} - Test Accuracy: {1:.3f} Train Accuracy: {2:.3f}'.format(
        epoch, test_acc, train_acc))

    model.train(True)
    return test_acc


def train_network(layerSize, num_of_ephocs, learning_rate_local, batch_size, indexes,
                  save_grads, data_sets_org, model_type, percent_of_train,
                  interval_accuracy_display, name, covn_net):
    """Train the network.

    PyTorch version with DataLoader and gradient collection.
    """
    device = DEVICE
    data_sets = data_shuffle(data_sets_org, percent_of_train)

    # Initialize storage
    ws = [None] * len(indexes)
    gradients = [None] * len(indexes)
    weights = [None] * len(indexes)
    loss_func_test = np.zeros(len(indexes))
    loss_func_train = np.zeros(len(indexes))
    test_prediction = np.zeros(len(indexes))
    train_prediction = np.zeros(len(indexes))

    # Get data dimensions
    input_size = data_sets_org.data.shape[1]
    num_of_classes = data_sets_org.labels.shape[1]
    batch_size = min(batch_size, data_sets.train.data.shape[0])

    # Create batch points for activation extraction
    batch_points_all = np.rint(
        np.arange(0, data_sets_org.data.shape[0] + 1, batch_size)
    ).astype(dtype=np.int32)
    if data_sets_org.data.shape[0] not in batch_points_all:
        batch_points_all = np.append(batch_points_all, [data_sets_org.data.shape[0]])

    # Build model
    model = build_model(model_type, layerSize, input_size, num_of_classes,
                        learning_rate_local, name, covn_net)
    model.to(device)

    # Create DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(data_sets.train.data),
        torch.FloatTensor(data_sets.train.labels)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_local)

    # Training loop
    k = 0  # Index into epoch snapshots
    acc_train_array = []

    for epoch in range(num_of_ephocs):
        epochs_grads = []
        current_weights = [[] for _ in range(len(model.weights_all))]

        # Extract activations at specified epochs (before training step)
        if epoch in indexes:
            ws[k] = extract_activity(model, batch_points_all, data_sets_org, device)

        # Print accuracy
        if np.mod(epoch, interval_accuracy_display) == 1 or interval_accuracy_display == 1:
            print_accuracy(model, data_sets, batch_size, epoch, acc_train_array, device)

        # Training step
        model.train(True)
        acc_train_array = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            # Cross-entropy loss (matching TF implementation)
            loss = -torch.mean(torch.sum(
                batch_y * torch.log(torch.clamp(outputs, 1e-50, 1.0)),
                dim=1
            ))

            loss.backward()

            # Calculate accuracy
            pred = outputs.argmax(dim=1)
            target = batch_y.argmax(dim=1)
            accuracy = (pred == target).float().mean().item()
            acc_train_array.append(accuracy)

            # Collect gradients and weights if needed
            if epoch in indexes and save_grads:
                batch_grads = [p.grad.clone().cpu().numpy()
                               for p in model.parameters() if p.grad is not None]
                epochs_grads.append(batch_grads)

                weights_local = [layer.weight.clone().cpu().numpy()
                                 for layer in model.hidden_layers_modules]
                for ii in range(len(current_weights)):
                    if ii < len(weights_local):
                        current_weights[ii].append(weights_local[ii])

            optimizer.step()

        # Store gradients and weights for this epoch
        if epoch in indexes:
            if save_grads:
                gradients[k] = epochs_grads
                current_weights_mean = []
                for ii in range(len(current_weights)):
                    if current_weights[ii]:
                        current_weights_mean.append(
                            np.mean(np.array(current_weights[ii]), axis=0)
                        )
                weights[k] = current_weights_mean
            k += 1

    # Build result dictionary
    network = {
        'ws': ws,
        'test_prediction': test_prediction,
        'train_prediction': train_prediction,
        'loss_test': loss_func_test,
        'loss_train': loss_func_train,
        'gradients': gradients,
        'weights': weights,
        'model': model,
    }

    return network
