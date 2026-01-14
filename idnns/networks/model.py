"""PyTorch model for Information Bottleneck analysis."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """A class that represents a neural network model for information analysis.

    This model supports dynamic layer construction and captures activations
    from all hidden layers for mutual information calculation.
    """

    def __init__(self, input_size, layerSize, num_of_classes, learning_rate_local=0.001,
                 save_file='', activation_function=0, cov_net=False):
        super().__init__()

        self.input_size = input_size
        self.layerSize = layerSize
        self.all_layer_sizes = np.copy(layerSize)
        self.all_layer_sizes = np.insert(self.all_layer_sizes, 0, input_size)
        self.num_of_classes = num_of_classes
        self._num_of_layers = len(layerSize) + 1
        self.learning_rate_local = learning_rate_local
        self._save_file = save_file
        self.covnet = cov_net

        # Select activation function
        if activation_function == 1:
            self.activation_function = F.relu
        elif activation_function == 2:
            self.activation_function = lambda x: x  # Linear
        else:
            self.activation_function = torch.tanh

        # Build layers dynamically
        self.hidden_layers_modules = nn.ModuleList()
        self.weights_all = []
        self.biases_all = []

        if cov_net == 1:
            self._build_cnn()
        elif cov_net == 2:
            self._build_mlp_variant()
        else:
            self._build_standard_mlp()

        # Storage for activation extraction (populated during forward pass)
        self._hidden_activations = []
        self._inputs = []

    def _build_standard_mlp(self):
        """Build standard MLP architecture."""
        # Hidden layers
        for i in range(1, len(self.all_layer_sizes)):
            in_features = int(self.all_layer_sizes[i - 1])
            out_features = int(self.all_layer_sizes[i])
            layer = nn.Linear(in_features, out_features)
            # Match TensorFlow initialization: truncated normal with std=1/sqrt(fan_in)
            nn.init.trunc_normal_(layer.weight, std=1.0 / np.sqrt(float(in_features)))
            nn.init.zeros_(layer.bias)
            self.hidden_layers_modules.append(layer)
            self.weights_all.append(layer.weight)
            self.biases_all.append(layer.bias)

        # Final output layer
        in_features = int(self.layerSize[-1])
        out_features = int(self.num_of_classes)
        final_layer = nn.Linear(in_features, out_features)
        nn.init.trunc_normal_(final_layer.weight, std=1.0 / np.sqrt(float(in_features)))
        nn.init.zeros_(final_layer.bias)
        self.hidden_layers_modules.append(final_layer)
        self.weights_all.append(final_layer.weight)
        self.biases_all.append(final_layer.bias)

    def _build_cnn(self):
        """Build CNN architecture for MNIST."""
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        self.pool = nn.MaxPool2d(2, 2)

        # FC layers
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)

        # Initialize with truncated normal
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
            if hasattr(layer, 'weight'):
                nn.init.trunc_normal_(layer.weight, std=0.1)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1)

    def _build_mlp_variant(self):
        """Build 2-layer MLP variant."""
        n_hidden_1 = int(self.layerSize[0])
        n_hidden_2 = int(self.layerSize[1])

        self.fc1 = nn.Linear(self.input_size, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc_out = nn.Linear(n_hidden_2, self.num_of_classes)

        # Random normal initialization (matches TF version)
        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)

    @property
    def num_of_layers(self):
        return self._num_of_layers

    @property
    def hidden_layers(self):
        """Return the captured hidden layer activations."""
        return self._hidden_activations

    @property
    def inputs(self):
        """Return the captured pre-activation inputs."""
        return self._inputs

    @property
    def save_file(self):
        return self._save_file

    def forward(self, x):
        """Forward pass with activation storage."""
        self._hidden_activations = []
        self._inputs = []

        if self.covnet == 1:
            return self._forward_cnn(x)
        elif self.covnet == 2:
            return self._forward_mlp_variant(x)
        else:
            return self._forward_standard_mlp(x)

    def _forward_standard_mlp(self, x):
        """Forward pass for standard MLP."""
        current = x

        # Hidden layers with activation function
        for i, layer in enumerate(self.hidden_layers_modules[:-1]):
            pre_activation = layer(current)
            self._inputs.append(pre_activation)
            current = self.activation_function(pre_activation)
            self._hidden_activations.append(current)

        # Final layer with softmax
        pre_activation = self.hidden_layers_modules[-1](current)
        self._inputs.append(pre_activation)
        output = F.softmax(pre_activation, dim=1)
        self._hidden_activations.append(output)

        return output

    def _forward_cnn(self, x, keep_prob=0.5):
        """Forward pass for CNN."""
        # Reshape for CNN if needed
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)

        self._hidden_activations.append(x.view(x.size(0), -1))  # Flattened input

        # Conv1 + Pool
        pre1 = self.conv1(x)
        h1 = F.relu(pre1)
        h1_pool = self.pool(h1)
        self._inputs.append(pre1)
        self._hidden_activations.append(h1_pool.view(h1_pool.size(0), -1))

        # Conv2 + Pool
        pre2 = self.conv2(h1_pool)
        h2 = F.relu(pre2)
        h2_pool = self.pool(h2)
        self._inputs.append(pre2)
        self._hidden_activations.append(h2_pool.view(h2_pool.size(0), -1))

        # Flatten
        h2_flat = h2_pool.view(-1, 7 * 7 * 64)

        # FC1 + Dropout
        pre_fc1 = self.fc1(h2_flat)
        h_fc1 = F.relu(pre_fc1)
        h_fc1 = self.dropout(h_fc1) if self.training else h_fc1
        self._inputs.append(pre_fc1)
        self._hidden_activations.append(h_fc1)

        # Output
        pre_out = self.fc2(h_fc1)
        output = F.softmax(pre_out, dim=1)
        self._inputs.append(pre_out)
        self._hidden_activations.append(output)

        return output

    def _forward_mlp_variant(self, x):
        """Forward pass for 2-layer MLP variant."""
        self._hidden_activations.append(x)

        # Layer 1
        pre1 = self.fc1(x)
        h1 = F.relu(pre1)
        self._inputs.append(pre1)
        self._hidden_activations.append(h1)

        # Layer 2
        pre2 = self.fc2(h1)
        h2 = F.relu(pre2)
        self._inputs.append(pre2)
        self._hidden_activations.append(h2)

        # Output
        pre_out = self.fc_out(h2)
        output = F.softmax(pre_out, dim=1)
        self._inputs.append(pre_out)
        self._hidden_activations.append(output)

        return output

    def get_hidden_activations(self, x):
        """Get all hidden layer activations for information calculation.

        Returns list of numpy arrays matching TensorFlow version's ws format.
        """
        was_training = self.training
        self.train(False)
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            _ = self.forward(x)
        self.train(was_training)
        return [h.cpu().numpy() if isinstance(h, torch.Tensor) else h
                for h in self._hidden_activations]

    def update_weights_and_calc_values_temp(self, d_w_i_j, layer_to_perturbe, i, j, X):
        """Update the weights of the given layer, calc the output and return it to the original values.

        This temporarily perturbs a single weight, gets output, then restores.
        Critical for gradient-based analysis in original code.
        """
        if self.covnet:
            raise NotImplementedError("Weight perturbation not implemented for CNN")

        layer = self.hidden_layers_modules[layer_to_perturbe]
        original_value = layer.weight.data[j, i].clone()

        with torch.no_grad():
            layer.weight.data[j, i] += d_w_i_j
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            _ = self.forward(X)
            layer_values = self._hidden_activations[layer_to_perturbe].cpu().numpy()
            layer.weight.data[j, i] = original_value

        return layer_values

    def update_weights(self, d_w0, layer_to_perturbe):
        """Update the weights' values of the given layer."""
        if self.covnet:
            raise NotImplementedError("Weight update not implemented for CNN")

        layer = self.hidden_layers_modules[layer_to_perturbe]
        with torch.no_grad():
            layer.weight.data += torch.FloatTensor(d_w0)

    def get_wights_size(self, layer_to_perturbe):
        """Return the size of the given layer (output_dim, input_dim)."""
        if self.covnet:
            raise NotImplementedError("get_wights_size not implemented for CNN")

        layer = self.hidden_layers_modules[layer_to_perturbe]
        return layer.weight.shape[0], layer.weight.shape[1]

    def get_layer_input(self, layer_to_perturbe, X):
        """Return the input of the given layer for the given data."""
        was_training = self.training
        self.train(False)
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            _ = self.forward(X)
        self.train(was_training)
        return self._inputs[layer_to_perturbe].cpu().numpy()

    def inference(self, data):
        """Return the prediction of the network with the given data."""
        was_training = self.training
        self.train(False)
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            pred = self.forward(data)
        self.train(was_training)
        return pred.cpu().numpy()

    def inference_default(self, data):
        """Return the prediction (same as inference in PyTorch)."""
        return self.inference(data)

    @property
    def cross_entropy(self):
        """Placeholder for cross entropy - computed in training loop."""
        return None

    @property
    def accuracy(self):
        """Placeholder for accuracy - computed in training loop."""
        return None

    @property
    def optimize(self):
        """Return an Adam optimizer for this model."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate_local)
