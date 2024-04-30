import matplotlib.pyplot as plt  # For plotting
import numpy as np  # Linear algebra library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# imports to visualize tree
from sklearn import tree as treeViz
import graphviz
from IPython.display import display

# Dataset provided in LaTeX format
# For this code to work, you would manually create the dataset in Python

# Create a DataFrame from the dataset
data = pd.DataFrame({
    'Outlook': ['rain', 'overcast', 'sunny', 'overcast', 'rain', 'overcast', 'sunny', 'overcast', 'sunny', 'rain',
                'rain', 'sunny', 'overcast'],
    'Wind': ['strong', 'weak', 'strong', 'strong', 'strong', 'strong', 'strong', 'weak', 'weak', 'weak', 'weak', 'weak',
             'weak'],
    'Temp': ['hot', 'hot', 'cool', 'cool', 'cool', 'hot', 'hot', 'hot', 'hot', 'hot', 'cool', 'cool', 'hot'],
    'Tennis': ['No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No']
})

data_encoded = pd.get_dummies(data.drop('Tennis', axis=1))
data_encoded['Tennis'] = data['Tennis'].apply(lambda x: 1 if x == 'Yes' else 0)

X = data_encoded.drop('Tennis', axis=1)
t = data_encoded['Tennis'].values

X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=0.3, random_state=1)

X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=1500 / 6500, random_state=1)

feature_names = X.columns.tolist()


def visualize_tree(model, feature_names, max_depth=5):
    dot_data = treeViz.export_graphviz(model,
                                       feature_names=feature_names,
                                       max_depth=max_depth,
                                       class_names=["No", "Yes"],
                                       filled=True,
                                       rounded=True)
    return display(graphviz.Source(dot_data))


tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)

tree.fit(X_train, t_train)

print("Training Accuracy:", tree.score(X_train, t_train))
print("Validation Accuracy:", tree.score(X_valid, t_valid))


def build_all_models(max_depths,
                     min_samples_split,
                     criterion,
                     X_train=X_train,
                     t_train=t_train,
                     X_valid=X_valid,
                     t_valid=t_valid):
    """
    Parameters:
        `max_depths` - A list of values representing the max_depth values to be
                       try as hyperparameter values
        `min_samples_split` - An list of values representing the min_samples_split
                       values to try as hyperpareameter values
        `criterion` -  A string; either "entropy" or "gini"

    Returns a dictionary, `out`, whose keys are the the hyperparameter choices, and whose values are
    the training and validation accuracies (via the `score()` method).
    In other words, out[(max_depth, min_samples_split)]['val'] = validation score and
                    out[(max_depth, min_samples_split)]['train'] = training score
    For that combination of (max_depth, min_samples_split) hyperparameters.
    """
    out = {}

    for d in max_depths:
        for s in min_samples_split:
            out[(d, s)] = {}
            # Create a DecisionTreeClassifier based on the given hyperparameters and fit it to the data
            tree = DecisionTreeClassifier(criterion=criterion, max_depth=d, min_samples_split=s)
            tree.fit(X_train, t_train)

            out[(d, s)]['val'] = tree.score(X_valid, t_valid)
            out[(d, s)]['train'] = tree.score(X_train, t_train)
    return out


criterions = ["entropy", "gini"]
max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

for criterion in criterions:
    print("\nUsing criterion {}".format(criterion))
    res = build_all_models(max_depths, min_samples_split, criterion)

    best_score = 0
    for d, s in res:
        if res[(d, s)]['val'] > best_score:
            best_score = res[(d, s)]['val']
            best_params = (d, s)

best_tree = DecisionTreeClassifier(criterion='entropy', max_depth=best_params[0], min_samples_split=best_params[1])

best_tree.fit(X_train, t_train)

test_score = best_tree.score(X_test, t_test)
print("Test Score:", test_score)

visualize_tree(tree, feature_names)

df = pd.DataFrame({
    'outlook_rain': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    'outlook_overcast': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    'outlook_sunny': [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'wind_strong': [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'wind_weak': [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    'temp_hot': [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1],
    'temp_cool': [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    'Tennis': ['No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No']
})

df['Tennis'] = df['Tennis'].map({'No': 0, 'Yes': 1})

X = df.drop('Tennis', axis=1)
t = df['Tennis']

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=1)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)

tree.fit(X_train, t_train)

plt.figure(figsize=(20, 10))
visualize_tree(tree, feature_names=X.columns)
plt.show()
