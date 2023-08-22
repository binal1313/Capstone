#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (replace 'data.csv' with your actual file)
data = pd.read_csv('Saffron Spreadsheet.csv')
data["Unit Price"] = pd.to_numeric(data["Unit Price"], errors="coerce")

# Define the number of simulations
num_simulations = 1000

# Define input variables and their distributions (you might need to customize this)
# For simplicity, let's assume Quantity follows a normal distribution and Unit Price is fixed
mean_quantity = data['Quantity'].mean()
std_quantity = data['Quantity'].std()

# Initialize an empty list to store simulated sales values
simulated_sales = []

# Perform Monte Carlo simulation
for _ in range(num_simulations):
    # Generate random values for input variables
    simulated_quantity = np.random.normal(mean_quantity, std_quantity)
    simulated_unit_price = data['Unit Price'].iloc[0]  # Assuming fixed unit price
    
    # Calculate simulated sales
    simulated_sales_value = simulated_quantity * simulated_unit_price
    simulated_sales.append(simulated_sales_value)


    
# Analyze and visualize results
simulated_sales = np.array(simulated_sales)
mean_sales = np.mean(simulated_sales)
std_sales = np.std(simulated_sales)


valid_simulated_sales = simulated_sales[~np.isnan(simulated_sales)]


# Create a histogram to visualize the distribution of simulated sales

print("Mean Simulated Sales:", mean_sales)
print("Standard Deviation of Simulated Sales:", std_sales)


# Create a histogram to visualize the distribution of simulated sales
#plt.hist(simulated_sales, bins=20, density=True, alpha=0.7)
#plt.xlabel("Sales")
#plt.ylabel("Density")
#plt.title("Distribution of Simulated Sales")
#plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (replace 'data.csv' with your actual file)
data = pd.read_csv('Saffron Spreadsheet.csv')

#data["Unit Price"] = pd.to_numeric(data["Unit Price"], errors="coerce")

# Define the number of simulations
num_simulations = 1000

# Define input variables and their distributions (you might need to customize this)
# For simplicity, let's assume Quantity follows a normal distribution and Unit Price is fixed
mean_quantity = data['Quantity'].mean()
std_quantity = max(data['Quantity'].std(), 1e-6)  # Ensure non-zero std for normal distribution

# Initialize an empty list to store simulated sales values
simulated_sales = []

# Perform Monte Carlo simulation
for _ in range(num_simulations):
    # Generate random values for input variables
    simulated_quantity = np.random.normal(mean_quantity, std_quantity)
    simulated_unit_price = data['Unit Price'].iloc[0]  # Assuming fixed unit price
    print("Simulated Quantity:", simulated_quantity)
    print("Simulated Unit Price:", simulated_unit_price)
    # Calculate simulated sales
    simulated_sales_value = simulated_quantity * simulated_unit_price
    simulated_sales.append(simulated_sales_value)

# Analyze and visualize results
simulated_sales = np.array(simulated_sales)
mean_sales = np.mean(simulated_sales)
std_sales = np.std(simulated_sales)

valid_simulated_sales = simulated_sales[~np.isnan(simulated_sales)]

# Create a histogram to visualize the distribution of simulated sales
print("Mean Simulated Sales:", mean_sales)
print("Standard Deviation of Simulated Sales:", std_sales)

#plt.hist(valid_simulated_sales, bins=20, density=True, alpha=0.7)
#plt.xlabel("Sales")
#plt.ylabel("Density")
#plt.title("Distribution of Simulated Sales")
#plt.show()

# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (replace 'data.csv' with your actual file)
data = pd.read_csv('Saffron Spreadsheet.csv')
data["Unit Price"] = pd.to_numeric(data["Unit Price"].str.replace('$', ''), errors="coerce")

# Define the number of simulations
num_simulations = 1000

# Define input variables and their distributions (you might need to customize this)
# For simplicity, let's assume Quantity follows a normal distribution and Unit Price is fixed
mean_quantity = data['Quantity'].mean()
std_quantity = max(data['Quantity'].std(), 1e-6)  # Ensure non-zero std for normal distribution

# Initialize an empty list to store simulated sales values
simulated_sales = []

# Perform Monte Carlo simulation
for _ in range(num_simulations):
    # Generate random values for input variables
    simulated_quantity = np.random.normal(mean_quantity, std_quantity)
    simulated_unit_price = data['Unit Price'].iloc[0]  # Assuming fixed unit price
    
    # Calculate simulated sales
    simulated_sales_value = simulated_quantity * simulated_unit_price
    simulated_sales.append(simulated_sales_value)

# Analyze and visualize results
simulated_sales = np.array(simulated_sales)
mean_sales = np.mean(simulated_sales)
std_sales = np.std(simulated_sales)

valid_simulated_sales = simulated_sales[~np.isnan(simulated_sales)]

# Create a histogram to visualize the distribution of simulated sales
print("Mean Simulated Sales:", mean_sales)
print("Standard Deviation of Simulated Sales:", std_sales)

plt.hist(valid_simulated_sales, bins=20, density=True, alpha=0.7)
plt.xlabel("Sales")
plt.ylabel("Density")
plt.title("Distribution of Simulated Sales")
plt.show()

# %%NET PROFIT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (replace 'data.csv' with your actual file)
data = pd.read_csv('Saffron Spreadsheet.csv')
data["Unit Price"] = pd.to_numeric(data["Unit Price"].str.replace('$', ''), errors="coerce")

# Define the number of simulations
num_simulations = 1000

# Define input variables and their distributions (you might need to customize this)
# For simplicity, let's assume Quantity follows a normal distribution and Unit Price is fixed
mean_quantity = data['Quantity'].mean()
std_quantity = max(data['Quantity'].std(), 1e-6)  # Ensure non-zero std for normal distribution

# Define cost factor
cost_factor = 0.6  # Example cost factor of 60%

# Initialize empty lists to store simulated sales and net profits
simulated_sales = []
simulated_net_profits = []

# Perform Monte Carlo simulation
for _ in range(num_simulations):
    # Generate random values for input variables
    simulated_quantity = np.random.normal(mean_quantity, std_quantity)
    simulated_unit_price = data['Unit Price'].iloc[0]  # Assuming fixed unit price
    
    # Calculate simulated sales
    simulated_sales_value = simulated_quantity * simulated_unit_price
    simulated_sales.append(simulated_sales_value)
    
    # Calculate simulated cost
    simulated_cost = simulated_quantity * cost_factor
    
    # Calculate simulated net profit
    simulated_net_profit = simulated_sales_value - simulated_cost
    simulated_net_profits.append(simulated_net_profit)

# Analyze and visualize results
simulated_net_profits = np.array(simulated_net_profits)
mean_net_profit = np.mean(simulated_net_profits)
std_net_profit = np.std(simulated_net_profits)

valid_simulated_net_profits = simulated_net_profits[~np.isnan(simulated_net_profits)]

# Create a histogram to visualize the distribution of simulated net profits
print("Mean Simulated Net Profit:", mean_net_profit)
print("Standard Deviation of Simulated Net Profit:", std_net_profit)

plt.hist(valid_simulated_net_profits, bins=20, density=True, alpha=0.7)
plt.xlabel("Net Profit")
plt.ylabel("Density")
plt.title("Distribution of Simulated Net Profits")
plt.show()

# %% With categorical field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (replace 'data.csv' with your actual file)
data = pd.read_csv('Saffron Spreadsheet.csv')
data["Unit Price"] = pd.to_numeric(data["Unit Price"].str.replace('$', ''), errors="coerce")

# Group data by product (Category or Sub Category) and calculate mean and std
product_grouped = data.groupby('Category').agg({'Quantity': 'mean', 'Unit Price': 'mean'}).reset_index()

# Define the number of simulations
num_simulations = 1000

# Initialize an empty dictionary to store simulated net profits for each product
simulated_net_profits = {}

# Perform Monte Carlo simulation for each product
for index, row in product_grouped.iterrows():
    simulated_net_profits[row['Category']] = []
    
    for _ in range(num_simulations):
        # Generate random values for simulated quantities based on the mean and std of Quantity
        simulated_quantity = np.random.normal(row['Quantity'], row['Quantity']*0.1)
        
        # Use the mean unit price for the specific product
        simulated_unit_price = row['Unit Price']
        
        # Calculate simulated net profit
        simulated_net_profit = simulated_quantity * simulated_unit_price
        simulated_net_profits[row['Category']].append(simulated_net_profit)

# Analyze and visualize results
for category, net_profits in simulated_net_profits.items():
    net_profits = np.array(net_profits)
    mean_net_profit = np.mean(net_profits)
    std_net_profit = np.std(net_profits)
    
    print(f"Product Category: {category}")
    print(f"Mean Simulated Net Profit: {mean_net_profit}")
    print(f"Standard Deviation of Simulated Net Profit: {std_net_profit}")
    print()

    # Create a histogram to visualize the distribution of simulated net profits
    plt.hist(net_profits, bins=20, density=True, alpha=0.7)
    plt.xlabel("Net Profit")
    plt.ylabel("Density")
    plt.title(f"Distribution of Simulated Net Profit - {category}")
    plt.show()

# %%
