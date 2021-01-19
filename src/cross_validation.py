import pandas as pd
from sklearn import model_selection


class CrossValidation():
    def __init__(self, dataframe, problem_type, target_cols, n_splits, shuffle=True, delimiter=None):
        '''
            List of problem_types = ["binary_classification", "multiclass_classification", "multilabel_classification", "singlecol_regression",
            "multicol_regression", "holdout"]
        '''
        self.dataframe = dataframe
        self.problem_type = problem_type
        self.target_cols = target_cols
        self.n_splits = n_splits
        if shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def split(self):
        num_targets = len(self.target_cols)
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            if num_targets!=1:
                raise Exception("Invalid number of columnns")
            
            target = self.target_cols[0]
            n_unique = self.dataframe[target].nunique()
            
            if n_unique==1:
                raise Exception("Only one column")
            elif n_unique>2 and self.problem_type=="binary_classification":
                raise Exception("Classes more than two for binary classification")
            elif n_unique<=2 and self.problem_type=="multiclass_classification":
                raise Exception("Classes less than or equal to two for multiclass classification")
            
            k_fold = model_selection.StratifiedKFold(n_splits=self.n_splits)
            self.dataframe["k_fold"] = -1
            for fold, (train_idx, val_idx) in enumerate(k_fold.split(self.dataframe, self.dataframe[target])):
                self.dataframe.loc[val_idx, "k_fold"] = fold    

        elif self.problem_type=="multilabel_classification":
            if num_targets!=1:
                raise Exception("Invalid number of target columns")
            target = self.target_cols[0]
            targets = self.dataframe[target].apply(lambda x:x.split(delimiter), axis=1)
            k_fold = model_selection.StratifiedKFold(n_splits=self.n_splits)
            self.dataframe["k_fold"] = -1
            for fold, (train_idx, val_idx) in enumerate(k_fold.split(self.dataframe, targets)):
                self.dataframe.loc[val_idx, "k_fold"] = fold
        
        elif self.problem_type in ["singlecol_regression", "multicol_regression"]:
            if num_targets!=1 and problem_type=="singlecol_regression":
                raise Exception("More targets than should be for single column regression")
            elif num_targets>2 and problem_type=="multicol_regression":
                raise Exception("Too less targets for multiple column regression")

            k_fold = model_selection.KFold(n_splits=self.n_splits)
            for fold, (train_idx, val_idx) in enumerate(k_fold.split(self.dataframe)):
                self.dataframe.loc[val_idx, "k_fold"] = fold

        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split('_')[1])
            train_len = len(dataframe) - int(len(dataframe)*holdout_percentage/100)
            self.dataframe.loc[:train_len, "k_fold"] = 0
            self.dataframe.loc[train_len:, "k_fold"] = 1
                
        else:
            raise Exception("Invalid problem type")

        return self.dataframe