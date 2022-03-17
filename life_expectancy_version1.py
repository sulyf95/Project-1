
# Import the packages that will be used

import pandas as pd
pd.set_option("max_colwidth", 20)
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")
from scipy.stats import pearsonr, ttest_ind

# Import and inspect the dataset 

df = pd.read_csv("all_data.csv")
df.dtypes
df.shape
df.head(5)
df.tail(5)

# Create a scatter graph to see the visual relationship between the two variables for each country

df = df.rename(
    columns={"Life expectancy at birth (years)": "Life_Expectancy"}
)  # column rename
count = 1
plt.figure(figsize=[15, 12])
for country in df.Country.unique():
    plt.subplot(2, 3, count)
    plt.scatter(
        x=df[df.Country == country]["GDP"],
        y=df[df.Country == country]["Life_Expectancy"],
    )
    plt.title(country, fontsize=14, fontweight="bold")
    plt.xlabel("GDP ($)", fontsize=14)
    plt.ylabel("Life Expectancy (Years)", fontsize=16)
    count += 1
plt.tight_layout()

# Correlation Coefficient analysis to see the strength of relationship between GDP and Life expectancy,
# in the six nations between 2000-2015

correlation_list = []
for country in df.Country.unique():
    corr_gdp_le, pval = pearsonr(
        df[df.Country == country]["GDP"], df[df.Country == country]["Life_Expectancy"]
    )
    coeff_ = round(corr_gdp_le, 2)
    correlation_list.append([country, coeff_])
correlation_list.sort(key=lambda x: x[1])
plt.figure(figsize=[15, 10])
sns.barplot(x=list(range(len(correlation_list))), y=[x[1] for x in correlation_list])
plt.title("The Realtionshp Between Life Expectancy and GDP", fontsize=20, fontweight="bold")
plt.xticks(
    list(range(len(correlation_list))),
    labels=[x[0] for x in correlation_list],
    fontsize=14,
)
plt.xlabel("Country", fontsize=14)
plt.ylabel("Correlation Coefficient", fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0.9, 1)
plt.show()
print(correlation_list)

# Splitting up the economies in small and large depending on if they are above or bellow the median.
# By putting the countries into categories, we can analyse the groups against each other

median_gdp = np.median(df["GDP"])
df["Size"] = df["GDP"].apply(
    lambda x: "Small Economy" if x < median_gdp else "Large Economy"
)
print("Large Economies:")
df[df.Size == "Large Economy"]
print("Small Economies:")
df[df.Size == "Small Economy"]

# Dropsing ['China', 2000] and ['Mexico', 2014], as they were outliers from their rest of the countries data
df = df.drop(index=[16, 62])

# Creating a visualistaion of boxplot and strippolt to see what the distribution of life expectancy,
# between the large and small economies are.

plt.figure(figsize=[15, 10])
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x="Size", y="Life_Expectancy")
plt.xlabel("Label", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Life Expectancy (in Years)", fontsize=14)
plt.yticks(fontsize=14)
plt.title(
    "Boxplot of Life Expectancy by Economy Size", fontsize=18, fontweight="bold"
)
plt.subplot(1, 2, 2)
sns.stripplot(data=df, x="Size", y="Life_Expectancy", jitter=0.05, hue="Country")
plt.xlabel("Label", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Life Expectancy (in Years)", fontsize=14)
plt.yticks(fontsize=14)
plt.title(
    "Stripplot of Life Expectancy by Economy Size", fontsize=18, fontweight="bold"
)
plt.tight_layout()

# Looking at the statistical difference in life expectancy using mean comaprison and two sample t-test.
# To see if there is a significant difference between the large and small economies life expectancy.

mean_large = np.mean(df["Life_Expectancy"][df.Size == "Large Economy"])
mean_small = np.mean(df["Life_Expectancy"][df.Size == "Small Economy"])
print("Mean Life Expectancy For Large Economies: {} years".format(round(mean_large, 2)))
print("Mean Life Expectancy For Small Economies: {} years".format(round(mean_small, 2)))
large_economies = df["Life_Expectancy"][df.Size == "Large Economy"]
small_economies = df["Life_Expectancy"][df.Size == "Small Economy"]
tstat, pval = ttest_ind(large_economies, small_economies, alternative="greater")
print("p-value: {}".format(pval))

# Seperating the dataset into large and small economies dataframes, and group them by year.
# Then find the mean from each year of the life expectancy in each dataframe.

df_large = df[df.Size == "Large Economy"]  # dataframe with Large Economies
df_small = df[df.Size == "Small Economy"]  # dataframe with Small Economies
df_large = df_large.groupby("Year").Life_Expectancy.mean().reset_index()
df_small = df_small.groupby("Year").Life_Expectancy.mean().reset_index()
print("Yearly Population Life Expectancy (Large Economies):")
df_large
print("Yearly Population Life Expectancy (Small Economies):")
df_small

# Collect the yearly mean life expectancy for both dataframes and store it in a list

yearly_pop_le_large = df_large["Life_Expectancy"]
yearly_pop_le_small = df_small["Life_Expectancy"]
gap = []
for x, y in zip(yearly_pop_le_large, yearly_pop_le_small):
    difference = x - y
    gap.append(difference)

# Plot a line graph to see the changes in life expectancy from year to year,
# between the large and small economies from 2000 to 2015

year_labels = df_large["Year"]
plt.figure(figsize=(15, 10))
plt.plot(year_labels, gap, linewidth=3)
plt.xlabel("Year", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Difference (Years)", fontsize=14)
plt.yticks(fontsize=14)
plt.title(
    "Population Life Expectancy Gap Between Large and Small Economies (2000 - 2015)",
    fontsize=16,
    fontweight="bold",
)