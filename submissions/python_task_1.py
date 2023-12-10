import pandas as pd
import numpy as np


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
     # Pivot the DataFrame to create the desired matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0
    car_matrix = car_matrix.fillna(0)

    # Set diagonal values to 0
    np.fill_diagonal(car_matrix.values, 0)

    return car_matrix
# Example usage
dataset_path = 'E:\Map Up India\Data Set and Intro\datasets\dataset-1.csv'
df = pd.read_csv(dataset_path)
result_matrix = generate_car_matrix(df)
print(result_matrix)


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    # Creating a new column 'car_type' based on the values in the 'car' column
    dataframe['car_type'] = pd.cut(dataframe['car'],
                                   bins=[-float('inf'), 15, 25, float('inf')],
                                   labels=['low', 'medium', 'high'],
                                   include_lowest=True, right=True)

    # Count of occurrences for each car_type category
    type_counts = dataframe['car_type'].value_counts().to_dict()

    # Sorting the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

file_path = 'E:\Map Up India\Data Set and Intro\datasets\dataset-1.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Call the function with the DataFrame
result = get_type_count(df)

# Display the result
print(result)

    
def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    # Calculate the mean value of the 'bus' column
    bus_mean = dataframe['bus'].mean()

    # Identifying indices where the 'bus' values are greater than twice the mean
    bus_indexes = dataframe[dataframe['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    sorted_bus_indexes = sorted(bus_indexes)

    return sorted_bus_indexes

file_path = 'E:\Map Up India\Data Set and Intro\datasets\dataset-1.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Call the function with the DataFrame
result = get_bus_indexes(df)

# Display the result
print(result)


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    # Calculating the average value of the 'truck' column for each unique 'route'
    route_avg_truck = dataframe.groupby('route')['truck'].mean()

    # Filter routes where the average truck value is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    sorted_routes = sorted(selected_routes)

    return sorted_routes

file_path = 'E:\Map Up India\Data Set and Intro\datasets\dataset-1.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Call the function with the DataFrame
result = filter_routes(df)

# Display the result
print(result)


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    # Create a copy of the DataFrame to avoid modifying the original
    modified_df = dataframe.copy()

    # Apply the specified logic to modify the values
    modified_df[dataframe > 20] *= 0.75
    modified_df[dataframe <= 20] *= 1.25

    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df
# Call the function with the DataFrame
modified_result = multiply_matrix(generate_car_matrix(df))

# Display the modified DataFrame
print(modified_result)


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Converting the timestamp columns to datetime format with explicit format
    dataframe['start_timestamp'] = pd.to_datetime(dataframe['startDay'] + ' ' + dataframe['startTime'], format='%y-%m-%d %I:%M:%S %p')
    dataframe['end_timestamp'] = pd.to_datetime(dataframe['endDay'] + ' ' + dataframe['endTime'], format='%y-%m-%d %I:%M:%S %p')

    # Calculate the time difference in minutes
    time_diff = (dataframe['end_timestamp'] - dataframe['start_timestamp']).dt.total_seconds() / 60

    # Check if each pair covers a full 24-hour period and spans all 7 days of the week
    completeness_check = (time_diff >= 24 * 60) & (dataframe['start_timestamp'].dt.dayofweek == 0) & (dataframe['end_timestamp'].dt.dayofweek == 6)

    # Create a multi-index series with (id, id_2)
    completeness_series = completeness_check.groupby([dataframe['id'], dataframe['id_2']]).all()

    return completeness_series

file_path = 'E:\Map Up India\Data Set and Intro\datasets\dataset-2.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Call the function with the DataFrame
result = verify_time_completeness(df)

# Display the result
print(result)

