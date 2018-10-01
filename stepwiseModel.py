import pandas as pd
import numpy as np
import statsmodels.api as smf

df = pd.read_csv('input.csv')

# Clean data
# Find the numeric data
# assign median to the missing values
missingdata = df.isnull().sum() / df.shape[0]
df = df.loc[:, missingdata < 0.7]
df = df._get_numeric_data()
df = df.fillna(data.median())

## Consider the column "Operating Income After Depreciation" as dependent variables Y; all others are independent variables X
Y = 'oiadp'


def forward_selected(data, response, sle=0.05):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data: pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    sle: significance level of a variable into the model

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept selected by forward selection
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
            score = smf.logit(formula, data).fit().pvalues[candidate]
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if best_new_score <= sle:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
        else:
            break
    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
    model = smf.logit(formula, data).fit()
    return model


def backward_selected(data, response, sls=0.01):
    """Linear model designed by backward selection.

    Parameters:
    -----------
    data: pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    sls: significance level of a variable to stay in the model

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept selected by backward selection
    """
    remaining = set(data.columns)
    remaining.remove(response)
    while remaining:
        formula = "{} ~ {} + 1".format(response, ' + '.join(remaining))
        scores = smf.logit(formula, data).fit().pvalues
        worst_new_score = scores.max()
        worst_candidate = scores.idxmax()
        if worst_new_score > sls:
            remaining.remove(worst_candidate)
        else:
            break
    formula = "{} ~ {} + 1".format(response, ' + '.join(remaining))
    model = smf.logit(formula, data).fit()
    return model


def stepwiseModel(data, response, sle=0.05, sls=0.01):
    """Linear model designed by stepwise selection.

    Parameters:
    -----------
    data: pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    sle: significance level of a variable into the model
    sls: significance level of a variable to stay in the model

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept selected by stepwise selection
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
            score = smf.logit(formula, data).fit().pvalues[candidate]
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if best_new_score <= sle:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
            scores = smf.logit(formula, data).fit().pvalues
            worst_new_score = scores.max()
            worst_candidate = scores.idxmax()
            if worst_new_score > sls:
                selected.remove(worst_candidate)
                remaining.add(worst_candidate)
                if best_candidate == worst_candidate: break
        else:
            break
    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
    model = smf.logit(formula, data).fit()
    return model

result = stepwiseModel(df, Y)
print('Selected columns by doing Stepwise Regression: ', result)
