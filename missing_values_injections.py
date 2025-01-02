from pandas import DataFrame
from numpy.random import seed as np_seed


RANDOM_STATE = 42
np_seed(RANDOM_STATE)

def inject_missing_values(
        data: DataFrame,
        columns: list[str],
        rows_severity: float | list[float]
 ) -> DataFrame:
    """
    Inject missing values into the data.
    
    Parameters
    ----------
    data : DataFrame
        The data to inject missing values into.
        
    columns : list[str]
        The columns to inject missing values into.

    rows_severity : float | list[float]
        The severity of the missing values to inject. If a float, the same severity is applied to all columns. If a list, the severity is applied to each column in order.
        
    Returns
    -------
    DataFrame
        The data with missing values injected.
    """
    assert isinstance(data, DataFrame), 'data must be a pandas DataFrame'
    if isinstance(rows_severity, (int, float)):
        rows_severity = [rows_severity] * len(columns)
    assert isinstance(rows_severity, list) and all(isinstance(severity, (int, float)) and 0 < severity < 1 for severity in rows_severity), 'rows_severity must be between 0 and 1'
    assert isinstance(columns, list) and all(column in data.columns for column in columns), 'not all columns are in the DataFrame'


    result_data = data.copy()

    for current_column, current_rows_severity in zip(columns, rows_severity):
        # Determine the number of missing rows
        missing_rows_count = int(len(data) * current_rows_severity)
        # Avoid cases where none/all rows are cleared
        if missing_rows_count == 0:
            missing_rows_count = 1
        if missing_rows_count == len(data):
            missing_rows_count -= 1
        # Inject missing values into the current column
        current_missing_rows_indices = result_data.sample(missing_rows_count).index
        result_data.loc[current_missing_rows_indices, current_column] = None

    return result_data