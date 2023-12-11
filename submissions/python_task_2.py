import pandas as pd
import networkx as nx
from datetime import time, datetime



def calculate_distance_matrix(df):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Creating a directed graph to represent toll locations and distances
    G = nx.DiGraph()

    # Add edges with distances to the graph
    for _, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
        G.add_edge(row['id_end'], row['id_start'], distance=row['distance'])  # Account for bidirectional distances

    # Calculate cumulative distances between toll locations
    distance_matrix = nx.floyd_warshall_numpy(G, weight='distance')

    # Create a DataFrame from the distance matrix
    distance_df = pd.DataFrame(distance_matrix, index=G.nodes, columns=G.nodes)

    return distance_df

# Example usage:
csv_file_path = './datasets/dataset-3.csv'
df = pd.read_csv(csv_file_path)
resulting_distance_matrix = calculate_distance_matrix(df)
print(resulting_distance_matrix)

def unroll_distance_matrix(distance_df):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        distance_df (pandas.DataFrame): Distance matrix

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initializing lists to store unrolled data
    id_start_list, id_end_list, distance_list = [], [], []

    # Iterate the rows and columns of the distance matrix
    for id_start in distance_df.index:
        for id_end in distance_df.columns:
            # Exclude same id_start to id_end pairs
            if id_start != id_end:
                distance = distance_df.loc[id_start, id_end]
                id_start_list.append(id_start)
                id_end_list.append(id_end)
                distance_list.append(distance)

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame({'id_start': id_start_list, 'id_end': id_end_list, 'distance': distance_list})

    return unrolled_df

# Example usage:
unrolled_distance_matrix = unroll_distance_matrix(resulting_distance_matrix)
print(unrolled_distance_matrix)

def find_ids_within_ten_percentage_threshold(df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Filter the DataFrame based on the reference value
    reference_df = df[df['id_start'] == reference_id]

    # Calculating the average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculating the threshold range (10% of the average distance)
    threshold = 0.1 * average_distance

    # Find the ids within the threshold range
    within_threshold_ids = df[(df['id_start'] != reference_id) & 
                              (df['distance'] >= (average_distance - threshold)) &
                              (df['distance'] <= (average_distance + threshold))]['id_start']

    # Sort and return the list of ids within the threshold range
    sorted_within_threshold_ids = sorted(within_threshold_ids.unique())

    return sorted_within_threshold_ids

# Example usage:
# Assuming unrolled_df is the DataFrame obtained from the previous function
resulting_matrix = calculate_distance_matrix('./datasets/dataset-3.csv')
unrolled_df = unroll_distance_matrix(resulting_matrix)

reference_value = unrolled_df['id_start'].iloc[0]  # Replace with the desired reference value
within_threshold_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)
print(within_threshold_ids)


def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    df = df.copy()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Iterate through each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        # Create a new column for each vehicle type with the calculated toll rates
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

# Example usage:
# Assuming unrolled_df is the DataFrame obtained from the previous function
resulting_df_with_rates = calculate_toll_rate(unrolled_df)
print(resulting_df_with_rates)  

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Make a copy of the input DataFrame to avoid modifying the original
    df = input_df.copy()

    # Define time ranges and discount factors
    time_ranges_weekday = [(time(0, 0, 0), time(10, 0, 0)),
                           (time(10, 0, 0), time(18, 0, 0)),
                           (time(18, 0, 0), time(23, 59, 59))]

    time_ranges_weekend = [(time(0, 0, 0), time(23, 59, 59))]

    discount_factors_weekday = [0.8, 1.2, 0.8]
    discount_factor_weekend = 0.7

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['end_day'] = df['start_time'] = df['end_time'] = ''

    # Iterate through each unique (id_start, id_end) pair
    for index, row in df.iterrows():
        # Iterate through each day of the week
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            for time_range_weekday, discount_factor in zip(time_ranges_weekday, discount_factors_weekday):
                start_time, end_time = time_range_weekday
                start_datetime = datetime.combine(datetime.min, start_time)
                end_datetime = datetime.combine(datetime.min, end_time)
                
                # Set values for the corresponding row in the DataFrame
                df.at[index, 'start_day'] = day
                df.at[index, 'end_day'] = day
                df.at[index, 'start_time'] = start_time
                df.at[index, 'end_time'] = end_time
                
                # Apply discount factor to each vehicle column individually
                for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                    df.at[index, vehicle_type] *= discount_factor

            for time_range_weekend in time_ranges_weekend:
                start_time, end_time = time_range_weekend
                start_datetime = datetime.combine(datetime.min, start_time)
                end_datetime = datetime.combine(datetime.min, end_time)
                
                # Set values for the corresponding row in the DataFrame
                df.at[index, 'start_day'] = day
                df.at[index, 'end_day'] = day
                df.at[index, 'start_time'] = start_time
                df.at[index, 'end_time'] = end_time
                
                # Apply discount factor to each vehicle column individually
                for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                    df.at[index, vehicle_type] *= discount_factor_weekend

    return df

# Example usage:
# Assuming resulting_df_with_rates is the DataFrame obtained from the previous function
resulting_df_with_rates = calculate_time_based_toll_rates(resulting_df_with_rates)
print(resulting_df_with_rates)