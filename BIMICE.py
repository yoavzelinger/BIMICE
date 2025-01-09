import numpy as np
import pandas as pd
from random import shuffle as shuffle_list

from sklearn.linear_model import LinearRegression

from graphlib import TopologicalSorter

from reparo import MICE
from causalnex.structure.notears import from_pandas as df_to_dependencies
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
	def _impute_column(X):
		model = LinearRegression()

		# Remove the rows with missing values
		X_clear_data = X[~np.isnan(X).any(axis=1)]

		# Slice out the column to predict and delete the column.
		y_clear_data = X_clear_data[:, -1]
		X_clear_data = np.delete(X_clear_data, -1, 1)

		# Train the model
		model.fit(X_clear_data, y_clear_data)

		# Predict the missing values
		X_data = np.delete(X, -1, 1)
		y = model.predict(X_data)

		return y

	def transform(self, X, iterations=10):
		# Get the relevant predictors for each column
		predicators_dict = BIMICE._get_inverse_dependencies(X)
		imputation_order = list(TopologicalSorter(predicators_dict).static_order())
		if not self.order_imputations:
			shuffle_list(imputation_order)
		if not self.filter_predicators:
			predicators_dict = {feature: list(range(X.shape[1])) for feature in range(X.shape[1])}

		features_missing_indices = [np.where(isnan_in_col) for isnan_in_col in np.isnan(X).T]

		X = X.copy()

		if not (self.order_imputations and self.filter_predicators):
			# Initial mean imputation
			features_means = np.nanmean(X, axis=0)
			missing_locations = np.where(np.isnan(X))
			X[missing_locations] = np.take(features_means, missing_locations[1])
		
		for i in range(iterations):
			for feature in imputation_order:
				feature_missing_indices = features_missing_indices[feature]
				predicators = predicators_dict[feature]
				if not predicators:
					X[np.isnan(X[:, feature]), feature] = np.nanmean(X[:, feature])
					if np.isnan(X[:, feature]).sum() > 0:
						raise ValueError(f'There are still NaN values in {feature} column (with no predicators)')
				else:
					current_sub_data = X[:, predicators + [feature]]
					X[feature_missing_indices, feature] = BIMICE._impute_column(current_sub_data)[feature_missing_indices]
					if np.isnan(X[:, feature]).sum() > 0:
						raise ValueError(f'There are still NaN values in {feature} column (with predicators)')
		return X, [len(predicators_dict[feature]) for feature in range(X.shape[1])]
	
	def fit_transform(self, X, iterations=2):
		return self.transform(X, iterations=iterations)