from typing import Dict, List
import re
import pandas as pd
import polyline
import math


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    i = 0
    
    while i < len(lst):
        group = []
        
        # Collect n elements in a group
        for j in range(i, min(i + n, len(lst))):
            group.append(lst[j])
        
        # Manually reverse the group and replace elements in the original list
        for k in range(len(group)):
            lst[i + k] = group[len(group) - 1 - k]
        
        i += n
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    
    # Sort the dictionary by the keys (lengths) and store it in a new variable
    sorted_result = dict(sorted(result.items()))

    # Assign the sorted result to a variable named `dict`
    dict_result = sorted_result  # Use `dict_result` to avoid conflict

    return dict_result


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened_dict = {}  # Renamed variable to avoid shadowing built-in dict

    def _flatten(current_dict: Dict, parent_key: str = ''):
        for key, value in current_dict.items():
            # Create new key with dot notation
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):  # If the value is a dictionary
                _flatten(value, new_key)
            elif isinstance(value, list):  # If the value is a list
                for index, item in enumerate(value):
                    if isinstance(item, dict):  # If the list item is a dictionary
                        _flatten(item, f"{new_key}[{index}]")
                    else:
                        flattened_dict[f"{new_key}[{index}]"] = item  # If it's a simple value
            else:
                flattened_dict[new_key] = value  # If it's a simple value

    _flatten(nested_dict)
    return flattened_dict  # Return the flattened dictionary

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current permutation
            return
        
        seen = set()  # Set to track used numbers at this level of recursion
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]  # Swap
            backtrack(start + 1)  # Recur
            nums[start], nums[i] = nums[i], nums[start]  # Swap back (backtrack)
    
    result = []  # Initialize result list
    nums.sort()  # Sort to handle duplicates
    backtrack(0)  # Start backtracking from index 0
    return result  # Return the list of unique permutations


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define regular expressions for each date format
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',   # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',   # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'   # yyyy.mm.dd
    ]

    # Combine all patterns into a single pattern
    combined_pattern = '|'.join(date_patterns)

    # Find all matches in the text
    matches = re.findall(combined_pattern, text)

    return matches


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude,
    and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    decoded_coords = polyline.decode(polyline_str)

    # Create lists to hold latitude, longitude, and distances
    latitudes = []
    longitudes = []
    distances = [0]  # Distance for the first point is 0

    # Populate latitude and longitude lists
    for i, (lat, lon) in enumerate(decoded_coords):
        latitudes.append(lat)
        longitudes.append(lon)

        # Calculate distance from the previous point if it's not the first point
        if i > 0:
            distance = haversine(decoded_coords[i - 1], (lat, lon))
            distances.append(distance)

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })

    return df

from typing import List

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then transform each element 
    to the sum of all elements in its original row and column in the rotated matrix, 
    excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the square matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]  # Create an empty n x n matrix
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Transform the rotated matrix
    final_matrix = [[0] * n for _ in range(n)]  # Create an empty n x n matrix
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the row and column excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
            
    return final_matrix

# Example usage
if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transformed_matrix = rotate_and_multiply_matrix(matrix)
    print(transformed_matrix)



def verify_timestamp_completeness(df):
    # Convert timestamp columns to datetime objects
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Check for null values in the new timestamp columns
    if df['start_timestamp'].isnull().any() or df['end_timestamp'].isnull().any():
        return "Invalid date/time format found in the input data."

    # Create a 24-hour time range for each day
    full_24_hours = pd.date_range('00:00:00', '23:59:59', freq='1S').time

    # Create a 7-day week
    full_week = set(pd.date_range('2024-01-01', periods=7).day_name())

    # Check if each timestamp covers a full 24-hour period and spans all 7 days
    completeness_check = (
        df.groupby(['id', 'id_2'])
        .apply(lambda group: (
            all(group['start_timestamp'].dt.time.isin(full_24_hours)) and
            all(group['end_timestamp'].dt.time.isin(full_24_hours)) and
            set(group['start_timestamp'].dt.day_name()) == full_week and
            set(group['end_timestamp'].dt.day_name()) == full_week
        ))
    )

    # Return the completeness check as a DataFrame
    return completeness_check.reset_index(name='is_complete')

# Read the dataset-2.csv file into a DataFrame
dataset_1 = pd.read_csv('dataset-1.csv')

# Call the function with the DataFrame
completeness_result = verify_timestamp_completeness(dataset_1)

# Display the result
print(completeness_result)

