import cudf
import sys
from ccfd.data.dataset import load_dataset_gpu

def display_n_records(filepath="cuml_oversampling_results.csv", num_records=5):
    """
    Reads the dataset from the CSV file and displays the first `num_records` rows.

    Args:
        filepath (str): Path to the CSV file.
        num_records (int): Number of records to display.
    """
    try:
        
        df = load_dataset_gpu("ccfd/data/creditcard.csv")
        if (df is None):
            return

        # Reset index to remove any extra index column
        df = df.reset_index(drop=True)

        # Display the results as a cuDF DataFrame

        # ✅ Display the specified number of records
        print(f"\n📊 Displaying First {num_records} Records:")
        print(df.head(num_records))

    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found. Please check the file path.")

if __name__ == "__main__":
    # ✅ Allow user to specify number of records via command-line argument
    num_records = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    display_n_records(num_records=num_records)


