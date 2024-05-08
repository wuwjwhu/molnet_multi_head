import pandas as pd

seed = 42

dir_prefix = f'datasets/seed_{seed}/'
# Replace these file names with your actual file paths
main_file = f'{dir_prefix}merged_training_set.csv'
supplement_files = [f'{dir_prefix}to_fill_training_set/bace.csv', f'{dir_prefix}to_fill_training_set/bbbp.csv', f'{dir_prefix}to_fill_training_set/clintox.csv', f'{dir_prefix}to_fill_training_set/sider.csv', f'{dir_prefix}to_fill_training_set/tox21.csv',
              f'{dir_prefix}to_fill_training_set/toxcast.csv', f'{dir_prefix}to_fill_training_set/esol.csv', f'{dir_prefix}to_fill_training_set/freesolv.csv', f'{dir_prefix}to_fill_training_set/lipo.csv']  # Add paths to your supplementary files

# Read the main file
main_df = pd.read_csv(main_file)

# Initialize a DataFrame to keep track of original (1) and complementary (0) values
original_values_df = pd.DataFrame(index=main_df.index, columns=main_df.columns)
original_values_df['smiles'] = 1  # 'smiles' column values are always original

# Process each supplementary file
for supplement_file in supplement_files:
    supplement_df = pd.read_csv(supplement_file)
    
    # Determine which labels are being supplemented from this file
    labels = supplement_df.columns.difference(['smiles'])
    
    # Merge the main DataFrame with this supplementary DataFrame
    merged_df = pd.merge(main_df, supplement_df, on='smiles', how='left', suffixes=('', '_supplement'))
    
    # Fill vacant values in the main DataFrame and update the tracking DataFrame
    for label in labels:
        # Find rows where the main DataFrame label is NaN and the supplement is not
        mask = main_df[label].isna() & merged_df[label + '_supplement'].notna()
        
        # Update the main DataFrame with supplemented values
        main_df.loc[mask, label] = merged_df.loc[mask, label + '_supplement']
        
        # Update the original/complementary tracking DataFrame
        original_values_df.loc[mask, label] = 0  # Mark as complementary (0)
        original_values_df.fillna(1, inplace=True)  # Mark remaining as original (1)

# Save the merged main DataFrame and the tracking DataFrame to new .csv files
main_df.to_csv(f'{dir_prefix}filled_training_set.csv', index=False)
original_values_df.to_csv(f'{dir_prefix}indicators.csv', index=False)

print("Merging and tracking operation completed.")
