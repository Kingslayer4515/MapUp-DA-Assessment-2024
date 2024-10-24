from typing import Dict, List,Any

import pandas as pd
import re
import polyline
import math


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    
    i = 0
    
    while i < len(lst):
        
        group_size = min(n, len(lst) - i)  
        for j in range(group_size // 2):
            
            lst[i + j], lst[i + group_size - j - 1] = lst[i + group_size - j - 1], lst[i + j]
        
        i += n
    
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    result = {}
    
    for string in lst:
        length = len(string) 
        if length not in result:
            result[length] = []  
        result[length].append(string) 
    
    
    sorted_result = dict(sorted(result.items()))
    
    return sorted_result


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:

    def flatten_helper(current_dict: Dict, parent_key: str = '') -> Dict[str, Any]:
        items = {}
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                
                items.update(flatten_helper(value, new_key))
            elif isinstance(value, list):
                
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        items.update(flatten_helper(item, f"{new_key}[{index}]"))
                    else:
                        items[f"{new_key}[{index}]"] = item
            else:
                
                items[new_key] = value
        return items

    if not nested_dict:
        return {}
    
    return flatten_helper(nested_dict)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  
            return
        seen = set()  
        for i in range(start, len(nums)):
            if nums[i] in seen:  
                continue
            seen.add(nums[i])  
            nums[start], nums[i] = nums[i], nums[start]  
            backtrack(start + 1)  
            nums[start], nums[i] = nums[i], nums[start]  

    nums.sort()  
    result = []
    backtrack(0)
    return result


def find_all_dates(text: str) -> List[str]:
    
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}\.\d{2}\.\d{2}\b'  
    ]
    
    combined_pattern = '|'.join(date_patterns)
    
    matches = re.findall(combined_pattern, text)
    
    return matches

def haversine(lat1, lon1, lat2, lon2):
    
    R = 6371000  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c 

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
   
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    df['distance'] = 0.0

    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)

    return df


def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
           
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix




def load_and_check_data(file_path: str) -> pd.DataFrame:
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Print the first few rows of the DataFrame
    print("Initial DataFrame:")
    print(df.head())
    
    # Check the data types of the relevant columns
    print("\nData Types:")
    print(df.dtypes)
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace and standardize case for days
    df['startDay'] = df['startDay'].str.strip().str.capitalize()
    df['endDay'] = df['endDay'].str.strip().str.capitalize()
    df['startTime'] = df['startTime'].str.strip()
    df['endTime'] = df['endTime'].str.strip()

    return df

def convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # Specify a base year for the conversion
    base_year = 2024  # You can choose any year

    # Combine date and time into datetime columns with a valid date format
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'] + f' {base_year}', errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'] + f' {base_year}', errors='coerce')

    return df

def time_check(df: pd.DataFrame) -> pd.Series:
    # Drop rows with NaT values in the datetime columns
    df.dropna(subset=['start_datetime', 'end_datetime'], inplace=True)

    # Check if DataFrame is empty after dropping NaNs
    if df.empty:
        print("DataFrame after dropping NaNs is empty.")
        return pd.Series(dtype=bool)

    # Set multi-index based on (id, id_2)
    df.set_index(['id', 'id_2'], inplace=True)

    # Define a function to check timestamps for each group
    def check_timestamps(group):
        start_time = group['start_datetime'].min()
        end_time = group['end_datetime'].max()

        covers_24_hours = (end_time - start_time) >= pd.Timedelta(hours=24)

        unique_days = pd.date_range(start=start_time.date(), end=end_time.date()).day_name()

        covers_all_days = set(unique_days) == {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}

        return not (covers_24_hours and covers_all_days)

    # Apply the function to each group and create the boolean Series
    result = df.groupby(level=['id', 'id_2']).apply(check_timestamps)

    return result

# Example usage:
file_path = '../MapUp-DA-Assessment-2024/datasets/dataset-1.csv'  # Give the csv file path after cloning in local
df = load_and_check_data(file_path)

# Clean the DataFrame
df = clean_data(df)

# Convert to datetime
df = convert_to_datetime(df)

# Run the time check
output = time_check(df)

# Display the output
print("Time Check Output:")
print(output)
