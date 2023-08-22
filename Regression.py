
#%%
import pandas as pd
import numpy as np


df = pd.read_csv("Traindata.csv")

# %%
e = pd.DataFrame(df)
e.describe(include = 'all')
df = df.dropna()
df.describe()

df
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data

# Select relevant columns
features = ['Sector', 'Category', 'Quantity', 'Payment Type', 'OrderDate']
target = 'Sales'

# Convert categorical columns to one-hot encoded features
categorical_columns = ['Sector', 'Category', 'Payment Type']
df_encoded = pd.get_dummies(df[features], columns=categorical_columns)

# Convert OrderDate to datetime and extract relevant features
df_encoded['OrderDate'] = pd.to_datetime(df_encoded['OrderDate'])
df_encoded['Year'] = df_encoded['OrderDate'].dt.year
df_encoded['Month'] = df_encoded['OrderDate'].dt.month
df_encoded['Day'] = df_encoded['OrderDate'].dt.day
df_encoded['DayOfWeek'] = df_encoded['OrderDate'].dt.dayofweek

# Drop original 'OrderDate' column
df_encoded.drop(['OrderDate'], axis=1, inplace=True)

# Split data into features (X) and target (y)
X = df_encoded
y = df[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# %%next step is to cross validate our model
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor  # Import your chosen model
from sklearn.externals import joblib

features = ['Sector', 'Category', 'Quantity', 'Payment Type', 'OrderDate']
target = 'Sales'

categorical_columns = ['Sector', 'Category', 'Payment Type']
df_encoded = pd.get_dummies(df[features], columns=categorical_columns)
df_encoded['OrderDate'] = pd.to_datetime(df_encoded['OrderDate'])
df_encoded['Year'] = df_encoded['OrderDate'].dt.year
df_encoded['Month'] = df_encoded['OrderDate'].dt.month
df_encoded['Day'] = df_encoded['OrderDate'].dt.day
df_encoded['DayOfWeek'] = df_encoded['OrderDate'].dt.dayofweek

df_encoded.drop(['OrderDate'], axis=1, inplace=True)

X = df_encoded
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse_scores = -scores  # Convert negative MSE back to positive
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

#%%saving model
import joblib

model_filename = 'trained_model.pkl'
joblib.dump(model, model_filename)


#%%

# Assuming y_test contains the actual sales values and y_pred contains the predicted sales values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.show()
#%%
import matplotlib.pyplot as plt

# Assuming y_test contains the actual sales values and y_pred contains the predicted sales values
plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Data Points')
plt.scatter(y_test, y_test, color='red', alpha=0.5, label='Perfect Prediction')  # Adding y_test points for comparison

plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.show()



#%%
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Sales', color='blue')
plt.plot(y_pred, label='Predicted Sales', color='orange')
plt.xlabel('Data Points')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
np.random.seed(0)
n_points = 100
y_test = np.random.randint(0, 100, n_points)
y_pred = y_test + np.random.normal(0, 10, n_points)

# Plot a subset of the data points
subset_indices = np.random.choice(range(n_points), size=20, replace=False)

plt.figure(figsize=(10, 6))
plt.plot(y_test[subset_indices], label='Actual Sales', marker='o')
plt.plot(y_pred[subset_indices], label='Predicted Sales', marker='x')
plt.xlabel('Data Points')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales (Subset)')
plt.legend()
plt.show()

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model from the .pkl file
loaded_model = joblib.load('trained_model.pkl')

new_data = pd.read_csv('Testdata.csv')  # Load new data

def preprocess_new_data(new_data):
    e = pd.DataFrame(new_data)
    e.describe(include = 'all')
    new_data =new_data.dropna()
    new_data.describe()
    features = ['Sector', 'Category', 'Quantity', 'Payment Type', 'OrderDate']
    target = 'Sales'
    
    # Convert categorical columns to one-hot encoded features
    categorical_columns = ['Sector', 'Category', 'Payment Type']
    new_data_encoded = pd.get_dummies(new_data[features], columns=categorical_columns)
    
    new_data_encoded['OrderDate'] = pd.to_datetime(new_data_encoded['OrderDate'])
    new_data_encoded['Year'] = new_data_encoded['OrderDate'].dt.year
    new_data_encoded['Month'] = new_data_encoded['OrderDate'].dt.month
    new_data_encoded['Day'] = new_data_encoded['OrderDate'].dt.day
    new_data_encoded['DayOfWeek'] = new_data_encoded['OrderDate'].dt.dayofweek

    new_data_encoded.drop(['OrderDate'], axis=1, inplace=True)
    # Standardize features
    scaler = StandardScaler()
    new_data_features = scaler.fit_transform(new_data_encoded)
    
    return new_data_features

# Preprocess new data

new_data_features = preprocess_new_data(new_data)

# Make predictions with new data

predictions = loaded_model.predict(new_data_features)

predicted_data = pd.DataFrame({'OrderID': new_data['OrderID'], 'PredictedSales': predictions})

predicted_data.to_csv('predicted_sales.csv', index=False)

# Evaluate predictions (if you have actual sales values for new data)
actual_sales = new_data['Sales']
# Assuming you have a column named 'Actual Sales'
mse = mean_squared_error(actual_sales, predictions)
r2 = r2_score(actual_sales, predictions)

# Visualize predictions (scatter plot)
import matplotlib.pyplot as plt

plt.scatter(actual_sales, predictions, color='blue', alpha=0.5)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales for New Data')
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
np.random.seed(0)
n_points = 100
y_test = np.random.randint(0, 100, n_points)
y_pred = y_test + np.random.normal(0, 10, n_points)

# Plot a subset of the data points
subset_indices = np.random.choice(range(n_points), size=20, replace=False)

plt.figure(figsize=(10, 6))
plt.plot(y_test[subset_indices], label='Actual Sales', marker='o')
plt.plot(y_pred[subset_indices], label='Predicted Sales', marker='x')
plt.xlabel('Data Points')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales (Subset)')
plt.legend()
plt.show()
#%%

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# %%
