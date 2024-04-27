from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, lambda_reg=0.1):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param lambda_reg: the value of the lambda regularization parameter
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_ids = np.array(data["user_id"])
    question_ids = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])

    x = theta[user_ids] - beta[question_ids]
    p_a = sigmoid(x)
    log_lklihood = is_correct * np.log(p_a) + (1 - is_correct) * np.log(1 - p_a)
    log_lklihood = np.sum(log_lklihood)

    l1_penalty = lambda_reg * (np.sum(np.abs(theta)) + np.sum(np.abs(beta)))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood + l1_penalty   # this has been changed to include the L2 regularization term


def update_theta_beta(data, lr, theta, beta, lambda_reg=0.1):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param lambda_reg: the value of the lambda regularization parameter
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_ids = np.array(data["user_id"])
    question_ids = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])

    x = theta[user_ids] - beta[question_ids]
    p_a = sigmoid(x)

    grad_theta = is_correct - p_a
    grad_beta = p_a - is_correct

    for u in np.unique(user_ids):
        theta_update = np.sum(grad_theta[user_ids == u]) - lambda_reg * np.sign(theta[u])
        theta[u] += lr * theta_update

    for q in np.unique(question_ids):
        beta_update = np.sum(grad_beta[question_ids == q]) - lambda_reg * np.sign(beta[q])
        beta[q] += lr * beta_update
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations, lambda_reg=0.1):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param lambda_reg: the value of the lambda regularization parameter
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(max(data["user_id"]) + 1)
    beta = np.zeros(max(data["question_id"]) + 1)

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta, beta, lambda_reg)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta, lambda_reg)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    user_ids = np.array(data["user_id"])
    question_ids = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])

    x = theta[user_ids] - beta[question_ids]
    p_a = sigmoid(x)
    predictions = p_a >= 0.5
    accuracy = np.mean(predictions == is_correct)
    return accuracy


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    learning_rate = 0.01
    lambda_reg = 0.001
    iterations = 300

    theta, beta, val_acc_lst = irt(train_data, val_data, learning_rate, iterations, lambda_reg)

    best_val_accuracy = max(val_acc_lst)
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    test_accuracy = evaluate(test_data, theta, beta)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    plt.plot(range(len(val_acc_lst)), val_acc_lst, label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j1, j2, j3 = 1, 2, 3

    theta_values = np.linspace(min(theta), max(theta), 100)
    p_correct_j1 = sigmoid(theta_values - beta[j1])
    p_correct_j2 = sigmoid(theta_values - beta[j2])
    p_correct_j3 = sigmoid(theta_values - beta[j3])

    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, p_correct_j1, label=f'Question {j1}')
    plt.plot(theta_values, p_correct_j2, label=f'Question {j2}')
    plt.plot(theta_values, p_correct_j3, label=f'Question {j3}')
    plt.xlabel('Ability (theta)')
    plt.ylabel('Probability of Correct Response')
    plt.title('Probability of Correct Response as a Function of Ability')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
