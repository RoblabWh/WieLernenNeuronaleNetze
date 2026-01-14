"""Mutual information estimation using variational methods - PyTorch version."""
import contextlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize

import idnns.information.entropy_estimators as ee


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def optimiaze_func(s, diff_mat, d, N):
    diff_mat1 = (1. / (np.sqrt(2. * np.pi) * (s ** 2) ** (d / 2.))) * np.exp(-diff_mat / (2. * s ** 2))
    np.fill_diagonal(diff_mat1, 0)
    diff_mat2 = (1. / (N - 1)) * np.sum(diff_mat1, axis=0)
    diff_mat3 = np.sum(np.log2(diff_mat2), axis=0)
    return -diff_mat3


def calc_all_sigams(data, sigmas):
    batchs = 128
    batch_points = np.rint(np.arange(0, data.shape[0] + 1, batchs)).astype(dtype=np.int32)
    I_XT = []
    num_of_rand = min(800, data.shape[1])

    for sigma in sigmas:
        I_XT_temp = 0
        for i in range(0, len(batch_points) - 1):
            new_data = data[batch_points[i]:batch_points[i + 1], :]
            rand_indexs = np.random.randint(0, new_data.shape[1], num_of_rand)
            new_data = new_data[:, :]
            N = new_data.shape[0]
            d = new_data.shape[1]
            diff_mat = np.linalg.norm(((new_data[:, np.newaxis, :] - new_data)), axis=2)
            s0 = 0.2
            res = minimize(optimiaze_func, s0, args=(diff_mat, d, N), method='nelder-mead',
                           options={'xtol': 1e-8, 'disp': False, 'maxiter': 6})
            eta = res.x
            diff_mat0 = - 0.5 * (diff_mat / (sigma ** 2 + eta ** 2))
            diff_mat1 = np.sum(np.exp(diff_mat0), axis=0)
            diff_mat2 = -(1.0 / N) * np.sum(np.log2((1.0 / N) * diff_mat1))
            I_XT_temp += diff_mat2 - d * np.log2((sigma ** 2) / (eta ** 2 + sigma ** 2))
        I_XT_temp /= len(batch_points)
        I_XT.append(I_XT_temp)

    sys.stdout.flush()
    return I_XT


class DecoderNetwork(nn.Module):
    """Small decoder network for estimating I(T;Y)."""

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc_out = nn.Linear(50, num_classes)

        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_out]:
            nn.init.trunc_normal_(layer.weight, std=1.0 / np.sqrt(float(layer.in_features)))
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc_out(x), dim=1)
        return x


def estimate_IY_by_network(data, labels, from_layer=0):
    """Estimate I(T;Y) by training a decoder network.

    Args:
        data: Layer activations
        labels: One-hot encoded labels
        from_layer: Layer index (skip training for deep layers)

    Returns:
        Tuple of (I_TY, accuracy)
    """
    device = DEVICE

    # Flatten input if needed
    if len(data.shape) > 2:
        input_size = int(np.prod(data.shape[1:]))
        data = data.reshape(data.shape[0], -1)
    else:
        input_size = data.shape[1]

    p_y_given_t_i = data
    acc_all = [0]

    if from_layer < 5:
        acc_all = []

        # Build decoder network
        model = DecoderNetwork(input_size, labels.shape[1])
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Training parameters
        num_of_epochs = 50
        batch_size = 51
        batch_points = np.rint(np.arange(0, data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
        if data.shape[0] not in batch_points:
            batch_points = np.append(batch_points, [data.shape[0]])

        # Convert to tensors
        data_tensor = torch.FloatTensor(data).to(device)
        labels_tensor = torch.FloatTensor(labels).to(device)

        # Training loop
        model.train()
        for epoch in range(num_of_epochs):
            for i in range(0, len(batch_points) - 1):
                batch_x = data_tensor[batch_points[i]:batch_points[i + 1]]
                batch_y = labels_tensor[batch_points[i]:batch_points[i + 1]]

                optimizer.zero_grad()
                outputs = model(batch_x)

                # Cross-entropy loss
                loss = -torch.mean(torch.sum(
                    batch_y * torch.log(torch.clamp(outputs, 1e-50, 1.0)),
                    dim=1
                ))

                loss.backward()
                optimizer.step()

        # Inference
        model.train(False)
        p_y_given_t_i = []
        batch_size = 256
        batch_points = np.rint(np.arange(0, data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
        if data.shape[0] not in batch_points:
            batch_points = np.append(batch_points, [data.shape[0]])

        with torch.no_grad():
            for i in range(0, len(batch_points) - 1):
                batch_x = data_tensor[batch_points[i]:batch_points[i + 1]]
                batch_y = labels_tensor[batch_points[i]:batch_points[i + 1]]

                outputs = model(batch_x)

                # Calculate accuracy
                pred = outputs.argmax(dim=1)
                target = batch_y.argmax(dim=1)
                acc = (pred == target).float().mean().item()
                acc_all.append(acc)

                if i == 0:
                    p_y_given_t_i = outputs.cpu().numpy()
                else:
                    p_y_given_t_i = np.concatenate(
                        (p_y_given_t_i, outputs.cpu().numpy()), axis=0
                    )

    # Calculate I(T;Y)
    max_indx = len(p_y_given_t_i)
    labels_cut = labels[:max_indx, :]
    true_label_index = np.argmax(labels_cut, 1)
    s = np.log2(p_y_given_t_i[np.arange(len(p_y_given_t_i)), true_label_index])
    I_TY = np.mean(s[np.isfinite(s)])
    PYs = np.sum(labels_cut, axis=0) / labels_cut.shape[0]
    Hy = np.nansum(-PYs * np.log2(PYs + np.spacing(1)))
    I_TY = Hy + I_TY
    I_TY = I_TY if I_TY >= 0 else 0
    acc = np.mean(acc_all)
    sys.stdout.flush()

    return I_TY, acc


def calc_varitional_information(data, labels, model_path, layer_numer, num_of_layers,
                                epoch_index, input_size, layerSize, sigma, pys, ks,
                                search_sigma=False, estimate_y_by_network=False):
    """Calculate estimation of the information using variational IB."""
    estimate_y_by_network = True
    data_x = data.reshape(data.shape[0], -1)

    if search_sigma:
        sigmas = np.linspace(0.2, 10, 20)
        sigmas = [0.2]
    else:
        sigmas = [sigma]

    if False:
        I_XT = calc_all_sigams(data_x, sigmas)
    else:
        I_XT = 0

    if estimate_y_by_network:
        I_TY, acc = estimate_IY_by_network(data, labels, from_layer=layer_numer)
    else:
        I_TY = 0

    with printoptions(precision=3, suppress=True, formatter={'float': '{: 0.3f}'.format}):
        print('[{0}:{1}] - I(X;T) - {2}, I(X;Y) - {3}, accuracy - {4}'.format(
            epoch_index, layer_numer, np.array(I_XT).flatten(), I_TY, acc))

    sys.stdout.flush()

    params = {}
    params['local_IXT'] = I_XT
    params['local_ITY'] = I_TY
    return params


def estimate_Information(Xs, Ys, Ts):
    """Estimation of the MI from missing data based on k-means clustering."""
    estimate_IXT = ee.mi(Xs, Ts)
    estimate_IYT = ee.mi(Ys, Ts)
    return estimate_IXT, estimate_IYT
