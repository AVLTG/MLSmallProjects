from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        self.sigmoid = nn.Sigmoid()

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        # Forward pass through the encoder layer with sigmoid activation
        encoded = self.sigmoid(self.g(inputs))

        # Forward pass through the decoder layer with sigmoid activation
        decoded = self.sigmoid(self.h(encoded))
        out = decoded

        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            # regularizer
            regularization_loss = 0.5 * lamb * model.get_weight_norm()
            loss += regularization_loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

def train_plot(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer. Plot training and validation objectives as a function of epoch.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: Tuple
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    valid_losses = []

    for epoch in range(0, num_epoch):
        # training
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            # regularizer
            regularization_loss = 0.5 * lamb * model.get_weight_norm()
            loss += regularization_loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        # validation
        model.eval()
        valid_loss = 0.

        for i, u in enumerate(valid_data["user_id"]):
            inputs = Variable(zero_train_data[u]).unsqueeze(0)
            target = inputs.clone()

            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[u].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            # regularizer
            regularization_loss = 0.5 * lamb * model.get_weight_norm()
            loss += regularization_loss
            valid_loss += loss.item()

        model.train()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    plt.title("Training and Validation Losses Per Epoch")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses[-1], valid_losses[-1]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    num_questions = zero_train_matrix.shape[1]
    # # Set model hyperparameters.
    for k in [10, 50, 100, 200, 500]:
        # Set optimization hyperparameters.
        for lr in [0.01, 0.05, 0.1]:
            for num_epoch in [10, 50, 100]:
                model = AutoEncoder(num_question=num_questions, k=k)
                lamb = None

                train(model, lr, lamb, train_matrix, zero_train_matrix,
                      valid_data, num_epoch)

                # evaluate model on validation set
                acc = evaluate(model, zero_train_matrix, valid_data)
                print(f"k: {k}, lr: {lr}, num_epoch: {num_epoch}, acc: {acc}")

    # chosen model
    k = 10
    lr = 0.01
    num_epoch = 100
    model = AutoEncoder(num_question=num_questions, k=k)
    lamb = None
    # train and plot
    train_plot(model, lr, lamb, train_matrix, zero_train_matrix,
               valid_data, num_epoch)

    # test accuracy
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Test Accuracy: {test_acc}")

    # tune regularization penalty
    for lamb in [0.001, 0.01, 0.1, 1]:
        model = AutoEncoder(num_question=num_questions, k=k)

        train(model, lr, lamb, train_matrix, zero_train_matrix,
              valid_data, num_epoch)

        # evaluate model on validation set
        acc = evaluate(model, zero_train_matrix, valid_data)
        print(f"lamb: {lamb} acc: {acc}")

    # chosen regularization penalty
    lamb = 0.001
    model = AutoEncoder(num_question=num_questions, k=k)
    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    valid_acc = evaluate(model, zero_train_matrix, valid_data)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Validation Accuracy: {valid_acc}, Test Accuracy: {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
