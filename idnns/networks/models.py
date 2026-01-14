"""PyTorch neural network architectures for Information Bottleneck analysis."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    """Multi-layer perceptron that captures intermediate activations."""

    def __init__(self, n_input, n_classes, n_hidden_1, n_hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc_out = nn.Linear(n_hidden_2, n_classes)

        # Random normal initialization (matches TF version)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias)
        nn.init.normal_(self.fc_out.weight)
        nn.init.normal_(self.fc_out.bias)

        self.hidden = []
        self.inputs = []

    def forward(self, x):
        self.hidden = [x]
        self.inputs = []

        # Layer 1 with ReLU
        layer_1_input = self.fc1(x)
        layer_1 = F.relu(layer_1_input)
        self.inputs.append(layer_1_input)
        self.hidden.append(layer_1)

        # Layer 2 with ReLU
        layer_2_input = self.fc2(layer_1)
        layer_2 = F.relu(layer_2_input)
        self.inputs.append(layer_2_input)
        self.hidden.append(layer_2)

        # Output layer with softmax
        output_input = self.fc_out(layer_2)
        y_output = F.softmax(output_input, dim=1)
        self.inputs.append(output_input)
        self.hidden.append(y_output)

        return y_output


class DeepCNN(nn.Module):
    """Deep CNN for MNIST classification with activation capture.

    This builds a CNN matching the original TensorFlow architecture:
    Conv1 (5x5, 32) -> Pool -> Conv2 (5x5, 64) -> Pool -> FC1 (1024) -> Dropout -> FC2 (10)
    """

    def __init__(self, dropout_prob=0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(dropout_prob)

        # Initialize weights with truncated normal (std=0.1)
        nn.init.trunc_normal_(self.conv1.weight, std=0.1)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.trunc_normal_(self.conv2.weight, std=0.1)
        nn.init.constant_(self.conv2.bias, 0.1)
        nn.init.trunc_normal_(self.fc1.weight, std=0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.trunc_normal_(self.fc2.weight, std=0.1)
        nn.init.constant_(self.fc2.bias, 0.1)

        self.hidden = []
        self.inputs = []
        self.keep_prob = None  # For compatibility

    def forward(self, x, keep_prob=None):
        """Forward pass capturing all intermediate activations.

        Args:
            x: Input tensor of shape (batch, 784) or (batch, 1, 28, 28)
            keep_prob: Dropout keep probability (for compatibility, uses module's dropout)
        """
        self.hidden = []
        self.inputs = []

        # Reshape if needed
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)

        self.hidden.append(x.view(x.size(0), -1))  # Flattened input

        # Conv1 + Pool
        input_conv1 = self.conv1(x)
        h_conv1 = F.relu(input_conv1)
        h_pool1 = self.pool(h_conv1)
        self.inputs.append(input_conv1)
        self.hidden.append(h_pool1)

        # Conv2 + Pool
        input_conv2 = self.conv2(h_pool1)
        h_conv2 = F.relu(input_conv2)
        h_pool2 = self.pool(h_conv2)
        self.inputs.append(input_conv2)
        self.hidden.append(h_pool2)

        # Flatten
        h_pool2_flat = h_pool2.view(-1, 7 * 7 * 64)

        # FC1 + Dropout
        input_fc1 = self.fc1(h_pool2_flat)
        h_fc1 = F.relu(input_fc1)
        h_fc1_drop = self.dropout(h_fc1) if self.training else h_fc1
        self.inputs.append(input_fc1)
        self.hidden.append(h_fc1_drop)

        # Output
        input_y_conv = self.fc2(h_fc1_drop)
        y_conv = F.softmax(input_y_conv, dim=1)
        self.inputs.append(input_y_conv)
        self.hidden.append(y_conv)

        return y_conv


def multi_layer_perceptron(x, n_input, n_classes, n_hidden_1, n_hidden_2):
    """Functional interface for multi-layer perceptron.

    This function creates and runs an MLP, returning outputs and captured activations.
    For compatibility with original TensorFlow API.

    Args:
        x: Input tensor
        n_input: Input size
        n_classes: Number of output classes
        n_hidden_1: First hidden layer size
        n_hidden_2: Second hidden layer size

    Returns:
        Tuple of (output, hidden_layers, inputs)
    """
    model = MultiLayerPerceptron(n_input, n_classes, n_hidden_1, n_hidden_2)
    if isinstance(x, torch.Tensor):
        output = model(x)
    else:
        output = model(torch.FloatTensor(x))
    return output, model.hidden, model.inputs


def deepnn(x):
    """Functional interface for deep CNN.

    This function creates and runs a CNN, returning outputs and captured activations.
    For compatibility with original TensorFlow API.

    Args:
        x: Input tensor of shape (batch, 784) for flattened MNIST images

    Returns:
        Tuple of (output, keep_prob_placeholder, hidden_layers, inputs)
        Note: keep_prob is None in PyTorch version (dropout handled by module)
    """
    model = DeepCNN()
    if isinstance(x, torch.Tensor):
        output = model(x)
    else:
        output = model(torch.FloatTensor(x))
    return output, None, model.hidden, model.inputs
