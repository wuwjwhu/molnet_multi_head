import pandas as pd
from sklearn.model_selection import train_test_split
import os

seed = 42

# Placeholder for the file names
file_names = ['datasets/bace.csv', 'datasets/bbbp.csv', 'datasets/clintox.csv', 'datasets/sider.csv', 'datasets/tox21.csv',
              'datasets/toxcast.csv', 'datasets/esol.csv', 'datasets/freesolv.csv', 'datasets/lipo.csv']

# Define an empty DataFrame to store all unique SMILES
unique_smiles = pd.DataFrame(columns=['smiles'])

# Store training data in a list for subsequent merging
training_data_frames = []

# Function to ensure directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Process each file
for idx, file_name in enumerate(file_names):
    df = pd.read_csv(file_name)
    
    file_name_only = os.path.basename(file_name)
    
    # Split into training and test set
    train_set, test_set = train_test_split(df, test_size=0.1, random_state=seed)
    
    # Ensure directories exist before saving
    ensure_dir(f'datasets/seed_{seed}/test_set/{file_name_only}')
    ensure_dir(f'datasets/seed_{seed}/finetune_set/{file_name_only}')
    
    # Save the test set with a unique name
    test_set.to_csv(f'datasets/seed_{seed}/test_set/{file_name_only}', index=False)
    
    # Save the train set as finetune set
    train_set.to_csv(f'datasets/seed_{seed}/finetune_set/{file_name_only}', index=False)
    
    # Append unique smiles from the training set to the comprehensive DataFrame
    unique_smiles = unique_smiles.merge(train_set[['smiles']].drop_duplicates(), on='smiles', how='outer')
    
    # Append the training set for later merging
    training_data_frames.append(train_set)

# Initialize the merged DataFrame with unique SMILES
merged_train_set = unique_smiles

# Merge each training set into the merged_train_set DataFrame
for idx, train_df in enumerate(training_data_frames):
    # Prepare the DataFrame to ensure all tags are in order and vacant values are handled
    merged_train_set = pd.merge(merged_train_set, train_df, on='smiles', how='left', suffixes=('', f'_file{idx+1}'))

# Ensure directory for merged training set
ensure_dir(f'datasets/seed_{seed}/merged_training_set.csv')

# Post-processing to clean up column names and handle vacant values
merged_train_set.to_csv(f'datasets/seed_{seed}/merged_training_set.csv', index=False)

file_names = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'esol', 'freesolv', 'lipo']

for idx, file_name in enumerate(file_names):
    # Replace these with your actual file paths
    csv_file_path = f'datasets/seed_{seed}/test_set/{file_name}.csv'
    csv_to_compare_path = f'datasets/seed_{seed}/merged_training_set.csv'
    output_csv_path = f'datasets/seed_{seed}/test_set/{file_name}_clean.csv'

    # Read the CSV files into DataFrames
    df_main = pd.read_csv(csv_file_path)
    df_compare = pd.read_csv(csv_to_compare_path)

    # Find unique "smiles" in the main DataFrame not present in the comparison DataFrame
    unique_smiles = set(df_main['smiles']) - set(df_compare['smiles'])

    # Filter the main DataFrame to keep only rows with unique "smiles"
    df_filtered = df_main[df_main['smiles'].isin(unique_smiles)]

    # Ensure directory for clean CSV
    ensure_dir(output_csv_path)

    # Save the filtered DataFrame to a new CSV file, preserving the header
    df_filtered.to_csv(output_csv_path, index=False)

    print("Filtered CSV saved to:", output_csv_path)

transfer_names = ['estrogen', 'hiv', 'metstab', 'qm7', 'qm8', 'qm9']

for idx, transfer_name in enumerate(transfer_names):
    # Replace these with your actual file paths
    csv_file_path = f'datasets/transfer_task/{transfer_name}.csv'
    csv_to_compare_path = f'datasets/seed_{seed}/merged_training_set.csv'
    output_csv_path = f'datasets/seed_{seed}/test_set/{transfer_name}_clean.csv'

    # Read the CSV files into DataFrames
    df_main = pd.read_csv(csv_file_path)
    df_compare = pd.read_csv(csv_to_compare_path)

    # Find unique "smiles" in the main DataFrame not present in the comparison DataFrame
    unique_smiles = set(df_main['smiles']) - set(df_compare['smiles'])

    # Filter the main DataFrame to keep only rows with unique "smiles"
    df_filtered = df_main[df_main['smiles'].isin(unique_smiles)]

    # Ensure directory for clean CSV
    ensure_dir(output_csv_path)

    # Save the filtered DataFrame to a new CSV file, preserving the header
    df_filtered.to_csv(output_csv_path, index=False)

    print("Filtered CSV saved to:", output_csv_path)
