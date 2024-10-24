import pandas as pd

from datetime import datetime, timedelta, time

def calculate_distance_matrix(df):
    # Create a sorted list of unique IDs from both id_start and id_end columns
    unique_ids = sorted(set(df['id_start']).union(df['id_end']))
    
    # Initialize the distance matrix with zeros
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    # Update the matrix with distances using .apply() for cleaner syntax
    def update_matrix(row):
        start_id = row['id_start']
        end_id = row['id_end']
        distance = row['distance']
        distance_matrix.at[start_id, end_id] += distance
        distance_matrix.at[end_id, start_id] += distance  # Ensure symmetry

    df.apply(update_matrix, axis=1)

    return distance_matrix

# Read the dataset-2.csv file into a DataFrame
df_dataset_2 = pd.read_csv('dataset-2.csv')

# Call the function with the DataFrame
distance_matrix = calculate_distance_matrix(df_dataset_2)

# Display the resulting distance matrix
print("Distance Matrix:")
print(distance_matrix)



import pandas as pd
import numpy as np

def unroll_distance_matrix(dist_matrix):
    # Validate that the distance matrix is symmetric
    if not (dist_matrix.index.equals(dist_matrix.columns)):
        raise ValueError("Distance matrix must be symmetric.")

    # Extract the upper triangular part of the matrix without the diagonal
    upper_triangular = dist_matrix.where(np.triu(np.ones(dist_matrix.shape), k=1).astype(bool))

    # Transform the upper triangular DataFrame into a long format
    unrolled_data = upper_triangular.stack().reset_index()

    # Rename columns to match the desired output
    unrolled_data.columns = ['id_start', 'id_end', 'distance']
    
    return unrolled_data

# Example usage (assuming dist_matrix is defined)
# Call the function with the DataFrame
unrolled_df = unroll_distance_matrix(distance_matrix)

# Display the resulting unrolled DataFrame
print("Unrolled Distance DataFrame:")
print(unrolled_df)



import pandas as pd

def find_ids_within_ten_percentage_threshold(dataframe, ref_value):
    # Extract rows for the specified reference ID
    ref_rows = dataframe[dataframe['id_start'] == ref_value]

    # Calculate the average distance for the reference ID
    avg_distance_ref = ref_rows['distance'].mean()

    # Determine the threshold limits (within 10% of the average distance)
    threshold_lower = avg_distance_ref * 0.9
    threshold_upper = avg_distance_ref * 1.1

    # Filter for distances within the defined threshold and get unique starting IDs
    ids_within_threshold = dataframe[
        (dataframe['distance'] >= threshold_lower) & 
        (dataframe['distance'] <= threshold_upper)
    ]['id_start'].unique()

    # Return a sorted list of unique IDs
    return sorted(ids_within_threshold)

# Example usage (assuming unrolled_df is defined)
# Reference value for demonstration purposes
reference_value = 100

# Call the function with the DataFrame and reference value
resulting_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)

# Display the sorted list of values within the 10% threshold
print("IDs within 10% threshold of reference value:", resulting_ids)



import pandas as pd

def compute_toll_fees(data_frame):
    # Define the fee rates for each type of vehicle
    fee_rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Create new columns in the DataFrame for each vehicle type with calculated toll fees
    for vehicle, rate in fee_rates.items():
        data_frame[f'{vehicle}_fee'] = data_frame['distance'] * rate

    return data_frame

# Example usage (assuming unrolled_df is defined)
# Call the function with the DataFrame
toll_rates_df = compute_toll_fees(unrolled_df)

# Display the resulting DataFrame with toll fees
print("DataFrame with Toll Fees:")
print(toll_rates_df)




def calculate_toll_rates_by_time(data_frame):
    # Define time ranges with corresponding discount factors for weekdays
    weekday_intervals = [
        (time(0, 0), time(10, 0), 0.8),   # Early morning discount
        (time(10, 0), time(18, 0), 1.2),  # Daytime rate
        (time(18, 0), time(23, 59, 59), 0.8)  # Evening discount
    ]
    
    # Set weekend discount factor
    weekend_discount_factor = 0.7
    
    # Initialize a DataFrame for storing results
    result_data = pd.DataFrame()

    # Iterate through each unique ('id', 'name', 'id_2') combination
    for (id_val, name_val, id_2_val), group in data_frame.groupby(['id', 'name', 'id_2']):
        for offset in range(7):  # Loop through each day of the week
            day_name = (datetime.today() + timedelta(days=offset)).strftime('%A')

            if offset < 5:  # Weekdays
                for start, end, discount in weekday_intervals:
                    start_dt = datetime.combine(datetime.today(), start) + timedelta(days=offset)
                    end_dt = datetime.combine(datetime.today(), end) + timedelta(days=offset)

                    # Update fees based on the applicable discount
                    group.loc[(group['timestamp'] >= start_dt) & (group['timestamp'] <= end_dt),
                               ['able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3', 'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3']] *= discount
                    
                    # Record the discount period
                    result_data = result_data.append({
                        'id': id_val,
                        'name': name_val,
                        'id_2': id_2_val,
                        'start_day': day_name,
                        'end_day': day_name,
                        'start_time': start,
                        'end_time': end
                    }, ignore_index=True)

            else:  # Weekends (Saturday and Sunday)
                group.loc[:, ['able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3', 'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3']] *= weekend_discount_factor
                
                result_data = result_data.append({
                    'id': id_val,
                    'name': name_val,
                    'id_2': id_2_val,
                    'start_day': day_name,
                    'end_day': day_name,
                    'start_time': time(0, 0),
                    'end_time': time(23, 59, 59)
                }, ignore_index=True)

    # Combine the original DataFrame with the results
    merged_result = pd.merge(data_frame, result_data, on=['id', 'name', 'id_2', 'start_day', 'end_day', 'start_time', 'end_time'], how='outer')

    return merged_result
#Q3 data

