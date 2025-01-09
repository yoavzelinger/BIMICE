from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from random import shuffle as shuffle_list

from graphlib import TopologicalSorter

from causalnex.structure.notears import from_pandas as df_to_dependencies
from causalnex.network import BayesianNetwork

class BIMICE(object):

	def __init__(self, order_imputations = True, filter_predicators = True):
		self.order_imputations = order_imputations
		self.filter_predicators = filter_predicators
	
	@staticmethod
	def _get_bayesian_network_edges(X: np.array) -> list[tuple[int]]:
		"""
		:param X: np.array
			input data
		:return: list of tuples
			list of edges. Each edge represents conditional dependence between the two nodes (directional).
		"""
		dependencies = df_to_dependencies(pd.DataFrame(X).dropna(), w_threshold=0.1)
		bayesian_network = BayesianNetwork(dependencies)
		return bayesian_network.edges
	
	@staticmethod
	def _get_inverse_dependencies(X: np.array) -> dict[int, list[int]]:
		"""
		:param X: np.array
			input data
		:return: dict
			inverse dependencies. Each key represents a node and the value is a list of its parents.
		"""
		return {0: [1, 2, 3], 1: [], 2: [1, 3], 3: [], 4: [0, 6, 7, 1, 2, 3], 5: [0, 6, 1, 3], 6: [0, 1, 2, 3], 7: [0, 5, 6, 1, 2, 3], 8: [0, 5, 6, 7, 1, 3]}
		dependencies_edges = BIMICE._get_bayesian_network_edges(X)
		inverse_dependencies = {independent: [] for independent in range(X.shape[1])}
		for dependent, independent in dependencies_edges:
			inverse_dependencies[independent].append(dependent)
		return inverse_dependencies
		
	@staticmethod
	def _get_mask(X):
		return np.isnan(X)

	def _process(self, X, column, model_class, y_missing_indices):
		# Remove the rows with missing values
		X_clear_data = np.delete(X, y_missing_indices, 0)

		# Slice out the column to predict and delete the column.
		y_clear_data = X_clear_data[:, column]
		X_clear_data = np.delete(X_clear_data, column, 1)

		# Instantiate the model
		model = model_class()

		# Fit the model
		model.fit(np.array(X_clear_data), y_clear_data)

		# Predict the missing values
		X_data = np.delete(X, column, 1)
		y = model.predict(X_data)
		return y[y_missing_indices]
	
	def transform(self, X, model_class=LinearRegression, iterations=10):
		# Get the relevant predictors for each column
		predicators_dict = BIMICE._get_inverse_dependencies(X)
		imputation_order = list(TopologicalSorter(predicators_dict).static_order())
		if not self.order_imputations:
			shuffle_list(imputation_order)
		if not self.filter_predicators:
			predicators_dict = {feature: list(range(X.shape[1])) for feature in range(X.shape[1])}

		features_missing_indices = np.isnan(X).T
		
		X = np.matrix(X)

		for i in range(iterations):
			for feature, feature_missing_indices in enumerate(features_missing_indices):
				predicators = predicators_dict[feature]
				if not predicators:
					X[np.isnan(X[:, feature]), feature] = np.nanmean(X[:, feature])
					if np.isnan(X[:, feature]).sum() > 0:
						raise ValueError(f'There are still NaN values in {feature} column (with no predicators)')
				else:
					X_sub = X[:, predicators + [feature]]
					X[feature_missing_indices, feature] = self._process(X_sub, feature, model_class, feature_missing_indices)
					if np.isnan(X[:, feature]).sum() > 0:
						raise ValueError(f'There are still NaN values in {feature} column (with predicators)')
		
		# Return X matrix with imputed values
		return X, [len(predicators_dict[feature]) for feature in range(X.shape[1])]
	
	def fit_transform(self, X, iterations=2):
		return self.transform(X, iterations=iterations)