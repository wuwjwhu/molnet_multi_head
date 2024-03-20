import pandas as pd

seed = 1

# file_names = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'esol', 'freesolv', 'lipo']

# for idx, file_name in enumerate(file_names):
#     # Replace these with your actual file paths
#     csv_file_path = f'datasets/random_{seed}/test_set/{file_name}.csv'
#     csv_to_compare_path = f'datasets/random_{seed}/merged_training_set.csv'
#     output_csv_path = f'datasets/random_{seed}/test_set/{file_name}_clean.csv'

#     # Read the CSV files into DataFrames
#     df_main = pd.read_csv(csv_file_path)
#     df_compare = pd.read_csv(csv_to_compare_path)

#     # Find unique "smiles" in the main DataFrame not present in the comparison DataFrame
#     unique_smiles = set(df_main['smiles']) - set(df_compare['smiles'])

#     # Filter the main DataFrame to keep only rows with unique "smiles"
#     df_filtered = df_main[df_main['smiles'].isin(unique_smiles)]

#     # Save the filtered DataFrame to a new CSV file, preserving the header
#     df_filtered.to_csv(output_csv_path, index=False)

#     print("Filtered CSV saved to:", output_csv_path)


transfer_names = ['estrogen', 'hiv', 'metstab', 'qm7', 'qm8', 'qm9']

for idx, transfer_name in enumerate(transfer_names):
    # Replace these with your actual file paths
    csv_file_path = f'datasets/transfer/{transfer_name}.csv'
    csv_to_compare_path = f'datasets/random_{seed}/merged_training_set.csv'
    output_csv_path = f'datasets/random_{seed}/test_set/{transfer_name}_clean.csv'

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