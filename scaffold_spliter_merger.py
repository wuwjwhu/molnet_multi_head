import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import train_test_split

# Random seed
seed = 1

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol)
    return scaffold

# Placeholder for the file names
file_names = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'esol', 'freesolv', 'lipo']

# Define an empty DataFrame to store all unique SMILES
unique_smiles = pd.DataFrame(columns=['smiles'])

# Store training data in a list for subsequent merging
training_data_frames = []

# Process each file
for idx, file_name in enumerate(file_names):
    # Load your dataset
    file_path = f'datasets/{file_name}.csv'
    df = pd.read_csv(file_path)

    # Generate scaffolds for each molecule
    df['scaffold'] = df['smiles'].apply(get_scaffold)  # Replace 'smiles_column' with the name of your SMILES column

    # Group by scaffold
    scaffold_groups = df.groupby('scaffold').groups
    scaffolds = list(scaffold_groups.keys())

    # Split scaffolds into training and test sets
    train_scaffolds, test_scaffolds = train_test_split(scaffolds, test_size=0.1, random_state=42)

    # Map scaffolds back to dataframe indices and split the dataset
    train_idx = sum((list(scaffold_groups[scaffold]) for scaffold in train_scaffolds), [])
    test_idx = sum((list(scaffold_groups[scaffold]) for scaffold in test_scaffolds), [])

    train_set = df.iloc[train_idx].drop(columns=['scaffold'])
    test_set = df.iloc[test_idx].drop(columns=['scaffold'])

    # Save the test set with a unique name
    test_set.to_csv(f'datasets/scaffold_{seed}/test_set/{file_name}.csv', index=False)
    
    # Save the train set as finetune set
    train_set.to_csv(f'datasets/scaffold_{seed}/finetune_set/{file_name}.csv', index=False)
    
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

# Post-processing to clean up column names and handle vacant values
# This is where you'd ensure all tags are retained in file order, and vacant values are left as NaN (or blank if exporting to CSV)
merged_train_set.to_csv(f'datasets/scaffold_{seed}/merged_training_set.csv', index=False)


# Reduplicate the testset
for idx, file_name in enumerate(file_names):
    # Replace these with your actual file paths
    csv_file_path = f'datasets/scaffold_{seed}/test_set/{file_name}.csv'
    csv_to_compare_path = f'datasets/scaffold_{seed}/merged_training_set.csv'
    output_csv_path = f'datasets/scaffold_{seed}/test_set/{file_name}_clean.csv'

    # Read the CSV files into DataFrames
    df_main = pd.read_csv(csv_file_path)
    df_compare = pd.read_csv(csv_to_compare_path)

    # Find unique "smiles" in the main DataFrame not present in the comparison DataFrame
    unique_smiles = set(df_main['smiles']) - set(df_compare['smiles'])

    # Filter the main DataFrame to keep only rows with unique "smiles"
    df_filtered = df_main[df_main['smiles'].isin(unique_smiles)]

    # Save the filtered DataFrame to a new CSV file, preserving the header
    df_filtered.to_csv(output_csv_path, index=False)

    print("Filtered CSV saved to:", output_csv_path)


transfer_names = ['estrogen', 'hiv', 'metstab', 'qm7', 'qm8', 'qm9']

for idx, transfer_name in enumerate(transfer_names):
    # Replace these with your actual file paths
    csv_file_path = f'datasets/transfer/{transfer_name}.csv'
    csv_to_compare_path = f'datasets/scaffold_{seed}/merged_training_set.csv'
    output_csv_path = f'datasets/scaffold_{seed}/test_set/{transfer_name}_clean.csv'

    # Read the CSV files into DataFrames
    df_main = pd.read_csv(csv_file_path)
    df_compare = pd.read_csv(csv_to_compare_path)

    # Find unique "smiles" in the main DataFrame not present in the comparison DataFrame
    unique_smiles = set(df_main['smiles']) - set(df_compare['smiles'])

    # Filter the main DataFrame to keep only rows with unique "smiles"
    df_filtered = df_main[df_main['smiles'].isin(unique_smiles)]

    # Save the filtered DataFrame to a new CSV file, preserving the header
    df_filtered.to_csv(output_csv_path, index=False)

    print("Filtered CSV saved to:", output_csv_path)