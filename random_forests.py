import numpy as np
import pandas as pd
import random

from pandas.api.types import is_numeric_dtype

class random_Forest_Classifier():

    def __init__(self, n_estimators=15, max_features=10, max_depth=10, min_samples_split=200):
        self.tree_ls = list()
        self.oob_ls = list()

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X_train, y_train):
        for i in range(self.n_estimators):
            X_bootstrap, y_bootstrap, X_oob, y_oob = self.draw_bootstrap(X_train, y_train)
            tree = self.build_tree(X_bootstrap, y_bootstrap, self.max_features, self.max_depth, self.min_samples_split)
            self.tree_ls.append(tree)
            oob_error = self.oob_score(tree, X_oob, y_oob)
            self.oob_ls.append(oob_error)
        print("OOB estimate: {:.2f}".format(np.mean(self.oob_ls)))

    # Gini Impurity
    def gini(self, p):
        return 1 - (p*p) - (1-p)*(1-p)

    # Calculates the amount of infomration that is gained from a potential split
    def information_gain(self, left, right):
        parent = left + right
        p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
        p_left = left.count(1) / len(left) if len(left) > 0 else 0
        p_right = right.count(1) / len(right) if len(right) > 0 else 0

        IG_p = self.gini(p_parent)
        IG_l = self.gini(p_left)
        IG_r = self.gini(p_right)

        # Information Gain = past_gini - split gini
        return IG_p \
            - len(left) / len(parent) * IG_l \
            - len(right) / len(parent) * IG_r

    def oob_score(self, tree, X_test, y_test):
        mis_label = 0
        for i in range(len(X_test)):
            pred = self.predict_tree(tree, X_test[i])
            if pred != y_test[i]:
                mis_label += 1
        return mis_label / len(X_test)

    # Bootstrapping to build randomised dataset for tree construction
    def draw_bootstrap(self, X_train, y_train):
        bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
        oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        X_oob = X_train[oob_indices]
        y_oob = y_train[oob_indices]
        return X_bootstrap, y_bootstrap, X_oob, y_oob
    
    def find_split_point(self, X_bootstrap, y_bootstrap, max_features):
        feature_ls = list()
        num_features = len(X_bootstrap[0])

        while len(feature_ls) <= max_features:
            feature_idx = random.sample(range(num_features), 1)
            if feature_idx not in feature_ls:
                feature_ls.extend(feature_idx)

            best_info_gain = -999
            node = None
            for feature_idx in feature_ls:
                split_points = np.unique(X_bootstrap[:,feature_idx])
                for split_point in split_points:
                    left_child = {'X_bootstrap': [], 'y_bootstrap': []}
                    right_child = {'X_bootstrap': [], 'y_bootstrap': []}

                    # split children for continuous variables
                    if type(split_point) in [int, float]:
                        for i, value in enumerate(X_bootstrap[:,feature_idx]):
                            if value <= split_point:
                                left_child['X_bootstrap'].append(X_bootstrap[i])
                                left_child['y_bootstrap'].append(y_bootstrap[i])
                            else:
                                right_child['X_bootstrap'].append(X_bootstrap[i])
                                right_child['y_bootstrap'].append(y_bootstrap[i])
                    # split children for categoric variables
                    else:
                        for i, value in enumerate(X_bootstrap[:,feature_idx]):
                            if value == split_point:
                                left_child['X_bootstrap'].append(X_bootstrap[i])
                                left_child['y_bootstrap'].append(y_bootstrap[i])
                            else:
                                right_child['X_bootstrap'].append(X_bootstrap[i])
                                right_child['y_bootstrap'].append(y_bootstrap[i])

                    split_info_gain = self.information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
                    if split_info_gain > best_info_gain:
                        best_info_gain = split_info_gain
                        left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
                        right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
                        node = {'information_gain': split_info_gain,
                                'left_child': left_child,
                                'right_child': right_child,
                                'split_point': split_point,
                                'feature_idx': feature_idx}


        return node


    def terminal_node(self, node):
        y_bootstrap = node['y_bootstrap']
        pred = max(y_bootstrap, key = y_bootstrap.count)
        return pred


    def split_node(self,node, max_features, min_samples_split, max_depth, depth):
        left_child = node['left_child']
        right_child = node['right_child']

        del(node['left_child'])
        del(node['right_child'])

        if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
            empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
            node['left_split'] = self.terminal_node(empty_child)
            node['right_split'] = self.terminal_node(empty_child)
            return

        if depth >= max_depth:
            node['left_split'] = self.terminal_node(left_child)
            node['right_split'] = self.terminal_node(right_child)
            return node

        if len(left_child['X_bootstrap']) <= min_samples_split:
            node['left_split'] = node['right_split'] = self.terminal_node(left_child)
        else:
            node['left_split'] = self.find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
            self.split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
        if len(right_child['X_bootstrap']) <= min_samples_split:
            node['right_split'] = node['left_split'] = self.terminal_node(right_child)
        else:
            node['right_split'] = self.find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
            self.split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

    def build_tree(self,X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
        root_node = self.find_split_point(X_bootstrap, y_bootstrap, max_features)
        self.split_node(root_node, max_features, min_samples_split, max_depth, 1)
        return root_node

    def random_forest(self, X_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
        tree_ls = list()
        oob_ls = list()
        for i in range(n_estimators):
            X_bootstrap, y_bootstrap, X_oob, y_oob = self.draw_bootstrap(X_train, y_train)
            tree = self.build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
            tree_ls.append(tree)
            oob_error = self.oob_score(tree, X_oob, y_oob)
            oob_ls.append(oob_error)
        print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
        return tree_ls

    def predict_tree(self,tree, X_test):
        feature_idx = tree['feature_idx']

        if X_test[feature_idx] <= tree['split_point']:
            if type(tree['left_split']) == dict:
                return self.predict_tree(tree['left_split'], X_test)
            else:
                value = tree['left_split']
                return value
        else:
            if type(tree['right_split']) == dict:
                return self.predict_tree(tree['right_split'], X_test)
            else:
                return tree['right_split']

    def predict(self, X_test):
        pred_ls = list()
        for i in range(X_test.shape[0]):
            ensemble_preds = [self.predict_tree(tree, X_test[i]) for tree in self.tree_ls]
            final_pred = max(ensemble_preds, key = ensemble_preds.count)
            pred_ls.append(final_pred)
        return np.array(pred_ls)





