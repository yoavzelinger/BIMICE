import numpy as np
import pandas as pd
from random import shuffle as shuffle_list

from graphlib import TopologicalSorter

from reparo import MICE
from causalnex.structure.notears import from_pandas as pd_to_dependencies
from causalnex.network import BayesianNetwork

class BIMICE(MICE):
    """
    Bayesian Improved Multiple Imputations by Chained Equations
    """

    def __init__(self, order_imputations = True, filter_predicators = True):
        self.order_imputations = order_imputations
        self.filter_predicators = filter_predicators
        super().__init__()

    @staticmethod
    def _get_bayesian_network_edges(X: np.array) -> list[tuple[int]]:
        """
        :param X: np.array
            input data
        :return: list of tuples
            list of edges. Each edge represents conditional dependence between the two nodes (directional).
        """
        dependencies = pd_to_dependencies(pd.DataFrame(X))
        bayesian_network = BayesianNetwork(dependencies)
        return bayesian_network.edges
    
    @staticmethod
    def _get_inverse_dependencies(self, X: np.array) -> dict[int, list[int]]:
        """
        :param X: np.array
            input data
        :return: dict
            inverse dependencies. Each key represents a node and the value is a list of its parents.
        """
        dependencies_edges = BIMICE._get_bayesian_network_edges(X)
        inverse_dependencies = {independent: [] for _, independent in range(X.shape[1])}
        for dependent, independent in dependencies_edges:
            inverse_dependencies[independent].append(dependent)
        return inverse_dependencies


    def transform(self, X: np.array, y: np.array = None, **fit_params):
        # Get the relevant predictors for each column
        predicators_dict = BIMICE._get_inverse_dependencies(X)
        imputation_order = list(TopologicalSorter(predicators_dict).static_order())
        if not self.order_imputations:
            shuffle_list(imputation_order)
        if not self.filter_predicators:
            predicators_dict = {feature: list(range(X.shape[1])) for feature in range(X.shape[1])}
        
        X = X.copy()
        for feature in range(X.shape[1]):
            predicators = predicators_dict[feature]
            current_sub_data = X[:, predicators + [feature]]
            # use super fit_transform
            current_sub_imputation = super().fit_transform(current_sub_data)
            X[:, feature] = current_sub_imputation[:, -1]
        return X