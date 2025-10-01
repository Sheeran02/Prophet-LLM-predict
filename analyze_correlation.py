import pandas as pd
from scipy.stats import spearmanr

# Load the CSV file
data = pd.read_csv('train_data.csv')

# Define the variables for correlation analysis
variables = ['CHWR_Temp', 'CHWS_Temp', 'Humidity', 'Temp', 'WetBulbTemp', 'Total_kW']

# Calculate Pearson correlation coefficients
correlation_results = {}
for var in variables:
    corr_coeff, _ = spearmanr(data['Total_kW'], data[var])
    correlation_results[var] = corr_coeff

# Print the correlation results
for var, coeff in correlation_results.items():
    print(f"Spearman correlation coefficient between Total_kW and {var}: {coeff}")
