import pandas as pd
import numpy as np
from datetime import time, timedelta


def calculate_distance_matrix(df) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Symmetric distance matrix with cumulative distances.
    """
    # Create a list of unique toll locations from the 'id_start' and 'id_end' columns
    toll_locations = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    # Create an empty DataFrame with toll locations as both rows and columns
    distance_matrix = pd.DataFrame(index=toll_locations, columns=toll_locations, data=float('inf'))
    
    # Set diagonal values to 0 (distance from a location to itself is 0)
    for loc in toll_locations:
        distance_matrix.loc[loc, loc] = 0
    
    # Fill in the matrix with known distances from the DataFrame
    for _, row in df.iterrows():
        loc_from = row['id_start']
        loc_to = row['id_end']
        distance = row['distance']
        
        # Since the matrix is symmetric, set both [From, To] and [To, From]
        distance_matrix.loc[loc_from, loc_to] = distance
        distance_matrix.loc[loc_to, loc_from] = distance
    
    # Use the Floyd-Warshall algorithm to compute cumulative distances between all toll locations
    for k in toll_locations:
        for i in toll_locations:
            for j in toll_locations:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
    
    return distance_matrix

# Load dataset
df = pd.read_csv('../MapUp-DA-Assessment-2024/datasets/dataset-2.csv') # give the correcct csv file path after cloning in local

# Generate distance matrix
distance_matrix = calculate_distance_matrix(df)
# print(distance_matrix)


def unroll_distance_matrix(df) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): The distance matrix.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initialize an empty list to hold the unrolled data
    unrolled_data = []

    # Get the indices (which are the unique ids) of the distance matrix
    ids = df.index.tolist()
    
    # Iterate through the distance matrix to extract the distances
    for i in ids:
        for j in ids:
            if i != j:  # Exclude pairs where id_start is the same as id_end
                distance = df.loc[i, j]
                # Append the data to the unrolled_data list
                unrolled_data.append({'id_start': i, 'id_end': j, 'distance': distance})
    
    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df

unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)

def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): The DataFrame containing id_start, id_end, and distance.
        reference_id (int): The reference ID for which to find the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for the reference_id
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    # Check if reference_id exists in the DataFrame
    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])
    
    average_reference_distance = reference_distances.mean()
    
    # Calculate the threshold values (10% above and below the average)
    lower_bound = average_reference_distance * 0.9
    upper_bound = average_reference_distance * 1.1
    
    # Calculate average distances for all id_start values
    average_distances = df.groupby('id_start')['distance'].mean().reset_index()
    average_distances.columns = ['id_start', 'average_distance']
    
    # Filter IDs within the 10% threshold of the reference ID's average distance
    filtered_ids = average_distances[
        (average_distances['average_distance'] >= lower_bound) &
        (average_distances['average_distance'] <= upper_bound)
    ]
    
    # Sort the result by id_start
    filtered_ids = filtered_ids.sort_values(by='id_start')
    
    return filtered_ids


result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id=1001402)
print(result_df)


def calculate_toll_rate(df) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing id_start, id_end, and distance.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for each vehicle type's toll rates.
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df

# Example usage:
# Assuming 'unrolled_df' is the DataFrame created in the previous step
toll_rate_df = calculate_toll_rate(unrolled_df)
print(toll_rate_df)


def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Define discount factors
    weekday_discounts = {
        '00:00:00-10:00:00': 0.8,
        '10:00:00-18:00:00': 1.2,
        '18:00:00-23:59:59': 0.8
    }
    weekend_discount = 0.7

    # Define time ranges
    time_ranges = {
        '00:00:00-10:00:00': (time(0, 0, 0), time(10, 0, 0)),
        '10:00:00-18:00:00': (time(10, 0, 0), time(18, 0, 0)),
        '18:00:00-23:59:59': (time(18, 0, 0), time(23, 59, 59))
    }

    # Define day ranges and discount associations
    periods = [
        # Weekdays periods
        ('Monday', 'Friday', weekday_discounts['00:00:00-10:00:00'], time(0, 0, 0), time(10, 0, 0)),
        ('Tuesday', 'Saturday', weekday_discounts['10:00:00-18:00:00'], time(10, 0, 0), time(18, 0, 0)),
        ('Wednesday', 'Sunday', weekday_discounts['18:00:00-23:59:59'], time(18, 0, 0), time(23, 59, 59)),
        # Weekend period (constant rate)
        ('Saturday', 'Sunday', weekend_discount, time(0, 0, 0), time(23, 59, 59))
    ]

    # Prepare a list to store the results
    result_data = []

    # Iterate over each unique (id_start, id_end) pair in the DataFrame
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        # Calculate base toll rates for each vehicle
        base_tolls = {vehicle: distance * rate for vehicle, rate in rate_coefficients.items()}

        # Apply each period and compute the adjusted tolls
        for start_day, end_day, discount_factor, start_time, end_time in periods:
            # Apply discount to each vehicle type
            adjusted_tolls = {vehicle: base_toll * discount_factor for vehicle, base_toll in base_tolls.items()}

            # Store the data
            result_data.append({
                'id_start': id_start,
                'id_end': id_end,
                'start_day': start_day,
                'start_time': start_time,
                'end_day': end_day,
                'end_time': end_time,
                **adjusted_tolls
            })

    # Convert the result data into a DataFrame
    result_df = pd.DataFrame(result_data)

    return result_df

# Example usage:
# Assuming 'unrolled_df' is the DataFrame created in the previous steps
time_based_toll_rate_df = calculate_time_based_toll_rates(unrolled_df)
print(time_based_toll_rate_df)
