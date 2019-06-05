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

class ColumnRenamer(BaseEstimator, TransformerMixin):

    """
    Renames columns with a dictionary
    """

    def __init__(self, rename_dict):
        self.rename_dict = rename_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.rename(columns=self.rename_dict)

        return X

class SubsetDuplicateRemover(BaseEstimator, TransformerMixin):

    """
    Drop duplicates based on a subset of columns
    """

    def __init__(self, subset_column_lst):
        self.subset_cols = subset_column_lst

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.drop_duplicates(subset=self.subset_cols)

        return X


class RowDropper(BaseEstimator, TransformerMixin):
    """
    Drops rows based on given conditions. Note: Most of the conditions apply only to text columns
    """

    def __init__(self, col_names, condition_equals=None, str_condition_contains=None, inverted=False, regex=False):

        self.col_names = col_names
        self.condition_equals = condition_equals
        self.inverted = inverted
        self.str_condition_contains = str_condition_contains
        self.regex = regex

    def fit(self, X=None, y=None):

        return self

    def transform(self, X, y=None):

        if self.condition_equals:

            if type(self.col_names) == list:

                for col in self.col_names:

                    if self.inverted:  # if inverted keep the records where condition matches

                        X[col] = X[col].str.strip()  # removing extra spaces so the condition is not missed

                        X = X[X[col] == self.condition_equals]
                    else:

                        X[col] = X[col].str.strip()  # removing extra spaces so the condition is not missed

                        X = X[X[col] != self.condition_equals]

            else:

                if self.inverted:  # if inverted keep the records where condition matches

                    X = X[X[self.col_names] == self.condition_equals]

                else:

                    X = X[(X[self.col_names] != self.condition_equals)]

        elif self.str_condition_contains:

            if type(self.col_names) == list:

                for col in self.col_names:

                    if self.inverted:  # if inverted keep the records where condition matches

                        X = X[(X[col].astype(str).str.contains(self.str_condition_contains, regex=self.regex))]
                    else:

                        # print(self.condition_equals)
                        X = X[~(X[col].astype(str).str.contains(self.str_condition_contains, regex=self.regex))]

            else:

                if self.inverted:  # if inverted keep the records where condition matches

                    X = X[(X[self.col_names].astype(str).str.contains(self.str_condition_contains, regex=self.regex))]

                else:

                    X = X[~(X[self.col_names].astype(str).str.contains(self.str_condition_contains, regex=self.regex))]

        return X


class NumericDTypeOptimizer(BaseEstimator, TransformerMixin):

    """
    Converts floats and ints to lower preicsion to make the DataFrame smaller for faster computations
    """

    def __init__(self, obj_cols_to_int=None, obj_cols_to_float=None):
        self.obj_cols_to_int = obj_cols_to_int
        self.obj_cols_to_float = obj_cols_to_float

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        if self.obj_cols_to_int is not None:

            for col in self.obj_cols_to_int:
                X[col] = X[col].astype("int16")

        if self.obj_cols_to_float is not None:

            for col in self.obj_cols_to_float:
                X[col] = X[col].astype("float32")

        return X

class EqualityChecker(BaseEstimator, TransformerMixin):

    """
    Checks if two column values are equal within the DataFrame. Can be used a feature.
    """

    def __init__(self, col_name_pairs, new_col_names, drop_originals=False):
        self.col_name_pairs = col_name_pairs
        self.new_col_names = new_col_names
        self.drop_originals = drop_originals

        assert len(self.col_name_pairs) == len(self.new_col_names), "Column names and column pairs in different numbers"

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):

        for idx, (col_a, col_b) in enumerate(self.col_name_pairs):
            new_col_name = self.new_col_names[idx]

            X[new_col_name] = (X[col_a] == X[col_b])

        if self.drop_originals:
            cols_to_be_deleted = list(set(sum(self.col_name_pairs, ())))

            X.drop(cols_to_be_deleted, axis=1, inplace=True)

            gc.collect()

        return X