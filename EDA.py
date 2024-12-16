import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv('clean_data.csv')

# Basic Information
print("Dataset Overview:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nSample Data:")
print(df.head())


# Histograms for numerical features
df.hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle('Distribution of Numerical Features', fontsize=16)
plt.show()

# Top 10 cuisines
top_cuisines = df['cuisines'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
plt.title('Top 10 Cuisines by Restaurant Count')
plt.xlabel('Number of Restaurants')
plt.ylabel('Cuisines')
plt.show()

# Top 10 cities
top_cities = df['city'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='coolwarm')
plt.title('Top 10 Cities by Restaurant Count')
plt.xlabel('Number of Restaurants')
plt.ylabel('City')
plt.show()

# Correlation between numeric features
# plt.figure(figsize=(12, 8))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='price_range', y='rating', data=df, palette='viridis')
plt.title('Rating Distribution Across Price Ranges')
plt.xlabel('Price Range')
plt.ylabel('Rating')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='average_cost_for_two', y='votes', hue='rating', size='rating', sizes=(20, 200), data=df, palette='coolwarm')
plt.title('Votes vs. Average Cost for Two')
plt.xlabel('Average Cost for Two')
plt.ylabel('Votes')
plt.show()

# Distribution of restaurants offering online delivery
delivery_counts = df['has_online_delivery'].value_counts()
labels = ['No Online Delivery', 'Online Delivery']

plt.figure(figsize=(8, 6))
plt.pie(delivery_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen'])
plt.title('Online Delivery Availability')
plt.show()

# Geographical Distribution of Restaurants
plt.figure(figsize=(12, 8))
sns.scatterplot(x='longitude', y='latitude', hue='rating', size='price_range', sizes=(20, 200), data=df, palette='coolwarm')
plt.title('Geographical Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Top cuisines by average cost
top_cuisines_avg_cost = df.groupby('cuisines')['average_cost_for_two'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_cuisines_avg_cost.values, y=top_cuisines_avg_cost.index, palette='plasma')
plt.title('Top 10 Cuisines by Average Cost')
plt.xlabel('Average Cost for Two')
plt.ylabel('Cuisines')
plt.show()
