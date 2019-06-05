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
