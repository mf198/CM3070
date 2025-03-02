# tests/test_dataset.py
import pandas as pd
import pytest
from ccfd.data.dataset import load_dataset, show_data_info

def test_load_dataset(tmp_path):
    """
    Test that the load_dataset function correctly reads a CSV file.
    """
    # Create a sample DataFrame
    data = {
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    }
    df_expected = pd.DataFrame(data)
    
    # Write the DataFrame to a temporary CSV file
    file_path = tmp_path / "sample.csv"
    df_expected.to_csv(file_path, index=False)
    
    # Use load_dataset to read the file
    df_loaded = load_dataset(str(file_path))
    
    # Assert that the loaded DataFrame matches the original DataFrame
    pd.testing.assert_frame_equal(df_expected, df_loaded)

def test_show_data_info(capsys):
    """
    Test that the show_data_info function prints the expected output.
    """
    # Create a simple DataFrame for testing
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    
    # Call show_data_info, which prints information about the DataFrame
    show_data_info(df)
    
    # Capture the output printed to stdout
    captured = capsys.readouterr().out
    
    # Verify that the output contains expected headers and column names
    assert "=== Dataset Info ===" in captured
    assert "=== Statistical Summary ===" in captured
    assert "A" in captured
    assert "B" in captured
