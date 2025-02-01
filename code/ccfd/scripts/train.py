from ccfd.data.dataset import load_dataset, show_data_info
from ccfd.data.preprocess import clean_dataset

# Load the dataset
df = load_dataset("ccfd/data/creditcard.csv")

# Show dataset info
show_data_info(df)

# Clean the dataset
df_clean = clean_dataset(df)

