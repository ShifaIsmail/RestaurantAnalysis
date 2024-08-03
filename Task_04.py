# TASK 4
# Location-based Analysis
# AIM =  Perform a geographical analysis of the restaurants in the dataset

# STEP1 = import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data exploration
data = pd.read_csv('Dataset.csv')


plt.figure(figsize=(10, 8))
plt.scatter(data['Longitude'], data['Latitude'], s=5)
plt.title("Restaurant Locations on Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# STEP3 =  Group restaurants by city or locality
city = data['City'].value_counts()
locality = data['Locality'].value_counts()

# STEP4 = Plot the number of restaurants in each city
plt.figure(figsize=(12, 6))
sns.barplot(x=city.index, y=city.values)
plt.xticks(rotation=90)
plt.title("Number of Restaurants by City")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()

# STEP5 = Plot the number of restaurants in each locality
plt.figure(figsize=(12, 6))
sns.barplot(x=locality.head(10).index, y=locality.head(10).values)
plt.xticks(rotation=90)
plt.title("Number of Restaurants by Locality (Top 10)")
plt.xlabel("Locality")
plt.ylabel("Count")
plt.show()

# STEP6 = Calculate statistics ranges by city or locality
rate_city = data.groupby('City')['Aggregate rating'].mean()
cuisine_city = data.groupby('City')['Cuisines'].count()
pricecity = data.groupby('City')['Average Cost for two'].mean()
