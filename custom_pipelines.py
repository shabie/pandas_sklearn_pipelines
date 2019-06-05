from sklearn.base import TransformerMixin, BaseEstimator


class NaNFiller(BaseEstimator, TransformerMixin):

    """
    Fill NaNs in a DataFrame
    """

    def __init__(self, columns, value=None, method=None, axis=None, limit=None):

        self.columns = columns
        self.value = value
        self.method = method
        self.axis = axis
        self.limit = limit

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):

        if type(self.columns) == list:

            for col in self.columns:
                X[col] = X[col].fillna(self.value, method=self.method, axis=self.axis, limit=self.limit)
        else:

            X[self.columns] = X[self.columns].fillna(self.value, method=self.method, axis=self.axis, limit=self.limit)

        return X

class SubstringReplacer(BaseEstimator, TransformerMixin):

    """
    Replace substring in multiple columns of a DataFrame
    """

    def __init__(self, columns, to_replace, replacement):

        self.to_replace = to_replace
        self.replacement = replacement
        self.columns = columns

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):

        if isinstance(self.columns, list):

            for col in self.columns:
                X[col] = X[col].astype(str).str.replace(self.to_replace, self.replacement)

        else:

            X[self.columns] = X[self.columns].astype(str).str.replace(self.to_replace, self.replacement)

        return X

class CharPadder(BaseEstimator, TransformerMixin):

    '''
    Class that's a wrapper around the pad function of pandas so it can be used in Pipelines
    '''

    def __init__(self, column, width, side='left', fillchar="0"):
        self.col_name = column
        self.width = width
        self.side = side.lower()
        self.fillchar = fillchar

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if type(self.col_name) == list:

            for col_name in self.col_name:
                X[col_name] = X[col_name].astype(str).str.pad(width=self.width, side=self.side, fillchar=self.fillchar)

        else:
            X[self.col_name] = X[self.col_name].astype(str).str.pad(width=self.width, side=self.side,
                                                                    fillchar=self.fillchar)

        return X


class FunctionApplyer(BaseEstimator, TransformerMixin):

    '''
    Class that's a wrapper around the apply function of pandas so it can be used in Pipelines
    '''

    def __init__(self, func, columns):

        self.func = func
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if type(self.columns) == list:  # I think may be isinstance works better

            for col in self.columns:
                X[col] = X[col].apply(self.func)
        else:

            X[self.columns] = X[self.columns].apply(self.func)

        return X