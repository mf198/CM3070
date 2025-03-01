import argparse
from ccfd.evaluation.plot_saved_curves import plot_curve_from_file

# Argument Parser
parser = argparse.ArgumentParser(description="Plot a saved curve from a results file.")
parser.add_argument("csv_file", type=str, help="Path to the saved CSV file.")

# Parse Arguments
args = parser.parse_args()

# Plot the Selected Curve
plot_curve_from_file(args.csv_file)
