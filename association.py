#%%

import pandas as pd

df = pd.read_csv("Saffron Spreadsheet.csv")

df
# %%
e = pd.DataFrame(df)
e.describe(include = 'all')

# %%
df = df.dropna()

df.describe()
# %%
df = df[df["Quantity"] >= 0]
df = df[df["Unit Price"] >= 0]

# %%
df["Sub Category"] = df["Sub Category"].str.strip()
df["Category"] = df["Category"].str.strip()
df
# %%
import matplotlib.pyplot as plt
import seaborn as sns

summary = (df.groupby("Category")
            .sum(numeric_only=True)
            .reset_index()
            .sort_values("Quantity", ascending= False)
            .head(20)
            )
summary

sns.barplot(
    data=summary,
    y="Category",
    x = "Quantity"
)

# %%
import pandas as pd
from datetime import datetime

# Assuming df is your DataFrame with the "OrderDate" column as strings
#df["OrderDate"] = pd.to_datetime(df["OrderDate"])  # Convert to datetime format

# Now add the "YearMonth" column
#df["YearMonth"] = df["OrderDate"].apply(
#    lambda dt: datetime(year=dt.year, month=dt.month, day=1)
#)


#df["GrossRevenue"]= df["Unit Price"] * df["Quantity"]

#summary = (
#    df.groupby("YearMonth")
#    .sum(numeric_only = True)
#    .reset_index()
#)
#summary


df["OrderDate"] = pd.to_datetime(df["OrderDate"])
# Convert OrderDate to YearMonth format
df["YearMonth"] = df["OrderDate"].apply(
    lambda dt: datetime(year=dt.year, month=dt.month, day=1)
)

# Group by Category and YearMonth and calculate total quantity sold
category_grouped = df.groupby(["Category", "YearMonth"]).sum()[["Quantity"]]

# Reset index for better presentation
category_grouped = category_grouped.reset_index()

print(category_grouped)


# %%
sns.lineplot(
    data= summary,
    x = "YearMonth",
    y = "GrossRevenue"
)


#%%
dataset = []
for inv, subset in df.groupby(["YearMonth", "Category"]):
    transactions = list(zip(subset["Category"], subset["Quantity"]))
    dataset.append(transactions)
dataset



# %% save df in pickle format with name "UK.pkl" for next lab activity
# we are only interested in InvoiceNo, StockCode, Description columns

#df[["Quantity", "Sub Category", "Category",'YearMonth']].to_pickle("Association.pkl")
# %%
#import pandas as pd

#df = pd.read_pickle("Association.pkl")
#df.head(20)

# %% apply apriori algorithm to find frequent items and association rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_itemsets, min_threshold=0.1)

rules

# %%
length = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets["length"] = length
frequent_itemsets

print ((frequent_itemsets["length"]>1).sum())
print ((frequent_itemsets["length"]>2).sum())
print ((frequent_itemsets["length"]>3).sum())

max_size = frequent_itemsets["length"].max()
print(
    frequent_itemsets[frequent_itemsets["length"] ==max_size]
    )

# %%
rules.sort_values("lift", ascending=False).head(10)


# %% scatterplot support vs confidence
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")


# %% scatterplot support vs lift
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["lift"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("lift")
plt.title("Support vs lift")


# %%Final

import pandas as pd

# Load your dataset
df = pd.read_csv('Saffron Spreadsheet1.csv')

# Convert OrderDate to datetime format
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# %%
# Group by OrderDate and Category, and sum the Quantity
aggregated_df = df.groupby(['OrderDate', 'Category']).agg({'Quantity': 'sum'}).reset_index()
aggregated_df

# %%
transaction_dataset = aggregated_df.groupby('OrderDate')['Category'].apply(list).reset_index(name='CategoryList')
transaction_dataset['TotalQuantity'] = aggregated_df.groupby('OrderDate')['Quantity'].sum().values

transaction_dataset

# %%most frequent itemset
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

te = TransactionEncoder()
te_ary = te.fit(transaction_dataset['CategoryList']).transform(transaction_dataset['CategoryList'])
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, min_threshold=0.1)


# Sort frequent_itemsets by support in descending order
sorted_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# Get the most frequent itemset (top one)
most_frequent_itemset = sorted_itemsets.iloc[0]['itemsets']
most_frequent_support = sorted_itemsets.iloc[0]['support']

print("Most Frequent Itemset:", most_frequent_itemset)
print("Support:", most_frequent_support)
output_filename = 'rules.csv'
rules.to_csv(output_filename, index=False)
print("Rules exported to:", output_filename)


#%%
sorted_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# Get the most frequent itemset (top one)
Second_frequent_itemset = sorted_itemsets.iloc[1]['itemsets']
Second_frequent_support = sorted_itemsets.iloc[1]['support']

print("Second Frequent Itemset:", Second_frequent_itemset)
print("Support:", Second_frequent_support)

sorted_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# Get the most frequent itemset (top one)
third_frequent_itemset = sorted_itemsets.iloc[2]['itemsets']
third_frequent_support = sorted_itemsets.iloc[2]['support']

print("third Frequent Itemset:", third_frequent_itemset)
print("Support:", third_frequent_support)
# %%
length = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets["length"] = length
frequent_itemsets

print ((frequent_itemsets["length"]>3).sum())
print ((frequent_itemsets["length"]>4).sum())
print ((frequent_itemsets["length"]>5).sum())

max_size = frequent_itemsets["length"].max()
print(
    frequent_itemsets[frequent_itemsets["length"] ==max_size]
    )


# %%
rules.sort_values("lift", ascending=False).head(10)
# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["lift"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("lift")
plt.title("Support vs lift")
# %%
