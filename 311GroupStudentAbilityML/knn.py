from sklearn.impute import KNNImputer

from group_11419.project.part_a.utils import sparse_matrix_evaluate, load_train_sparse, load_valid_csv, load_public_test_csv
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    print('knn_impute_by_user')
    best_k = 0
    best_accuracy = 0
    accuracies_by_user = []
    for k in [1, 6, 11, 16, 21, 26]:
        accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracies_by_user.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    print(f'k* = {best_k}')
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print(f'Test Accuracy: {test_accuracy} with knn_impute_by_user with k = {best_k}')

    print('knn_impute_by_item')
    best_k = 0
    best_accuracy = 0
    accuracies_by_item = []
    for k in [1, 6, 11, 16, 21, 26]:
        accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        accuracies_by_item.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    print(f'k* = {best_k}')
    test_accuracy = knn_impute_by_item(sparse_matrix, test_data, best_k)
    print(f'Test Accuracy: {test_accuracy} with knn_impute_by_item with k = {best_k}')

    plt.plot([1, 6, 11, 16, 21, 26], accuracies_by_user)
    plt.title('Validation Accuracy By User')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot([1, 6, 11, 16, 21, 26], accuracies_by_item)
    plt.title('Validation Accuracy By Item')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
