import pandas as pd
import numpy as np


def generate_car_matrix(df)->pd.DataFrame:
    # Write your logic here
     # Pivot the DataFrame to create the desired matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0
    car_matrix = car_matrix.fillna(0)

    # Set diagonal values to 0
    np.fill_diagonal(car_matrix.values, 0)

    return car_matrix
# Example usage
dataset_path = './datasets/dataset-1.csv'
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
    df['car_type'] = pd.cut(df['car'],
                                   bins=[-float('inf'), 15, 25, float('inf')],
                                   labels=['low', 'medium', 'high'],
                                   include_lowest=True, right=True)

    # Count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sorting the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

file_path = './datasets/dataset-1.csv'
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
    bus_mean = df['bus'].mean()

    # Identifying indices where the 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    sorted_bus_indexes = sorted(bus_indexes)

    return sorted_bus_indexes

file_path = './datasets/dataset-1.csv'
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
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average truck value is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    sorted_routes = sorted(selected_routes)

    return sorted_routes

file_path = './datasets/dataset-1.csv'
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
    modified_df = df.copy()

    # Apply the specified logic to modify the values
    modified_df[df > 20] *= 0.75
    modified_df[df <= 20] *= 1.25

    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df
# Call the function with the DataFrame
modified_result = multiply_matrix(generate_car_matrix(df))

# Display the modified DataFrame
print(modified_result)


def multiply_matrix(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Create a copy of the DataFrame to avoid modifying the original
    modified_df = df.copy()

    # Apply the specified logic to modify the values
    modified_df[df > 20] *= 0.75
    modified_df[df <= 20] *= 1.25

    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

# Assuming 'result_dataframe' is the DataFrame obtained from Question 1
# Replace 'result_dataframe' with the actual DataFrame you have

# Call the function with the DataFrame
modified_result = multiply_matrix(generate_car_matrix(df))

# Display the modified DataFrame
print(modified_result)
