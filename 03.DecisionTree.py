import numpy as np
from numpy import random as rd
import pandas as pd

original_data = pd.read_csv("compustat_annual_2000_2017_with link information.csv")
#original_data.shape

Y_Column = "oibdp"

# remove missing Ys
training_data = original_data[original_data[Y_Column].notnull()]
#training_data.shape

# imputing numerical columns
training_data = training_data.fillna(training_data.median())

# imputing non-missing columns
training_data = training_data.fillna("")

list(training_data.isnull().sum())

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_columns = training_data.select_dtypes(include=numerics)

char_columns = training_data.select_dtypes(exclude=numerics)
#"acctstd" in char_columns.columns

Y_Data = training_data[Y_Column]
X_Data = training_data.drop(Y_Column,axis=1)

header = X_Data.columns

class Question:
    """A Question is used to partition a dataset.

    The class gets column name, and a value.
    """

    def __init__(self, column_name, value):
        self.column = column_name
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if self.column in numeric_columns:
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if self.column in numeric_columns:
            condition = ">="
        return "Is %s %s %s?" % (
            self.column, condition, str(self.value))

q = Question("LPERMCO", 3000)
#q

def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the weighted variance."""

    
    Y_Data = rows[Y_Column]
    X_Data = rows.drop(Y_Column, axis=1)
    
    best_variance = Y_Data.var()  # keep track of the best variance
    best_question = None  # keep train of the feature / value that produced it

    for col in header:  # for each feature

        values = X_Data[col].sort_values().unique()  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows = Y_Data[question.match(X_Data)]
            false_rows = Y_Data[~question.match(X_Data)]

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the weighted variance from this split
            split_variance = len(true_rows)*1.0/len(Y_Data)*1.0*true_rows.var()+len(false_rows)*1.0/len(Y_Data)*1.0*false_rows.var()


            if split_variance <= best_variance:
                best_variance, best_question = split_variance, question

    return best_variance, best_question

#######
# Demo:
# Test with a small dataset
test_data = training_data[["acctstd","LIID",Y_Column]]
Y_Data = test_data[Y_Column]
X_Data = test_data.drop(Y_Column,axis=1)

header = X_Data.columns

best_variance, best_question = find_best_split(test_data)
#best_variance

#######

class Leaf:
    """A Leaf node classifies data.

    It stores the node prediction which is the average of observations.
    """

    def __init__(self, rows):
        self.predictions = rows[Y_Column].mean()

class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

max_depth = 2

def build_tree(rows, i):
    """Builds the tree.


    """
    if i == max_depth:
        return Leaf(rows)
    
    i += 1

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the variance,
    # and return the question that produces the lowest variance.
    variance, question = find_best_split(rows)

    # Base case: no further variance gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if variance == rows[Y_Column].var():
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows = rows[question.match(rows)]
    false_rows = rows[~question.match(rows)]

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, i)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, i)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")
    
my_tree = build_tree(test_data, 0)
print_tree(my_tree)

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

h = test_data[test_data.iloc[:,0] == "DI"].iloc[0,:]
#training_data.columns
#training_data["LIID"].unique()

#h
#######
# Demo:

classify(h, my_tree)
#######

our_data = (test_data.iloc[0:1000,:])
for i in range (len(our_data)):
    print(classify(our_data.iloc[i,:], my_tree))
