import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Random 
seed = 1

# Placeholder for the file names
file_names = ['datasets/bace.csv', 'datasets/bbbp.csv', 'datasets/clintox.csv', 'datasets/sider.csv', 'datasets/tox21.csv',
              'datasets/toxcast.csv', 'datasets/esol.csv', 'datasets/freesolv.csv', 'datasets/lipo.csv']
# file_names = ['demo/demo1.csv', 'demo/demo2.csv', 'demo/demo3.csv']

# Define an empty DataFrame to store all unique SMILES
unique_smiles = pd.DataFrame(columns=['smiles'])

# Store training data in a list for subsequent merging
training_data_frames = []

# Process each file
for idx, file_name in enumerate(file_names):
    df = pd.read_csv(file_name)
    
    file_name_only = os.path.basename(file_name)
    
    # Split into training and test set
    train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
    
    # Save the test set with a unique name
    test_set.to_csv(f'datasets/random_{seed}/test_set/{file_name_only}', index=False)
    
    # Save the train set as finetune set
    train_set.to_csv(f'datasets/random_{seed}/finetune_set/{file_name_only}', index=False)
    
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
merged_train_set.to_csv(f'datasets/random_{seed}/merged_training_set.csv', index=False)
