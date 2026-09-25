#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction Project
# **Goal:** We want to build an AI model that looks at house details (like area, rooms) and predicts its final `SalePrice`.
# 
# ### Phase 1: What we have done so far?
# 1. **Created a Project Folder:** Set up the environment using `uv`.
# 2. **Downloaded Data:** Got the official `train.csv` and `test.csv` files from Kaggle.
#    - **Train Data:** Contains house details AND the final price (AI learns from this).
#    - **Test Data:** Contains house details but NO price (AI will guess this later).
# 3. **Loaded Libraries & Data:** Brought our data into Python using Pandas.

# In[1]:


#===================================
# STEP 1: IMPORTING TOOLS (LIBRARIES)
#===================================

# import pandas: This acts like Microsoft Excel for python. It helps us read , clean  and view data tables
# Import Matplotlib: This tool give us a blank canvas to draw basic charts & graphs
import matplotlib.pyplot as plt

#Import Numpy: This act like as a fast calculator for doing math on big numbers
import numpy as np  # noqa: F401, RUF100
import pandas as pd

# Import Seaborn: This tool helps us create more attractive and informative statistical graphics
import seaborn as sns

#Tell pandas not to hide any column. we want to see all 80+ columns on our screen
pd.set_option('display.max_columns', None)

print("All tools are ready")


# In[2]:


# ==========================================
# STEP 2: LOADING THE CSV FILES
# ==========================================

train_path = r"E:\new_projects\House_pridiction_price_project\data\train.csv"
test_path  = r"E:\new_projects\House_pridiction_price_project\data\test.csv"


df_train = pd.read_csv(train_path)
df_test =  pd.read_csv(test_path)

print(f"Shape of df_train: {df_train.shape}")
print(f"Shape of df_test:  {df_test.shape}")


# In[3]:


# Display all columns without hiding any (Safe & Recommended)
pd.set_option("display.max_columns", None)


# In[4]:


df_train.head() # Display the first few rows of the training dataset


# In[5]:


df_test.head() 
# Display the first few rows of the test dataset


# # Data Preprocessing Kya Hai Aur Kyun Zaroori Hai?
# 
# Sochein aap biryani bana rahe hain. Agar chawal kharab hain ya namak zyada hai, toh biryani bhi kharab banegi. Bilkul waise hi, agar data ganda hai toh model bhi ghalat jawab dega. Preprocessing ka matlab hai:
# 
# - Ghalat ya adhoori information hatana (missing values)
# - Cheezon ko sahi format mein lana (numbers vs text)
# - Faaltu columns hata dena jo kaam ke nahi
# 
# Kaunse Features Price Pe Asar Daalte Hain?
# 
# 1. OverallQual -> Ghar ki quality (1-10) 10 = Bohot acha, 1 = Bohot kharab
# 
# 2. GrLivArea   -> Ghar ka size (sq ft) 2000 sq ft > 1000 sq ft = Mehnga
# 
# 3. YearBuilt   -> Kab bana  2020 ka ghar > 1950 ka ghar
# 
# 4. Neighborhood  -> Ilaqa -> NridgHt mehnga, MeadowV sasta
# 
# 5. GarageCars  ->   Kitni gaariyan aayein  3 cars > 1 car
# 
# 6. FullBath   ->    Bathrooms  -> Zyada bathroom = Zyada price
# 
# 7. TotalBsmtSF  ->  Basement ka size  -> Bada basement = Mehnga
# 
# 8. KitchenQual  ->  Kitchen kaisa hai  -> Ex (Excellent) > Po (Poor) 

# # 🔑 Essential DataFrame Inspection Methods (The 80/20 Rule)
# 
# When beginning Exploratory Data Analysis (EDA) in **pandas**, these four commands provide the majority of the initial structural insights needed to understand your dataset.
# 
# ## 1. `df.shape` — Dataset Dimensions
# * **Type:** Attribute (no parentheses `()`)
# * **Purpose:** Returns a tuple indicating the number of rows and columns: `(rows, columns)`.
# * **Interpretation:** An output like `(1460, 81)` means the dataset contains **1,460 rows (records)** and **81 columns (features)**.
# 
# ## 2. `df.columns` — Feature Names
# * **Type:** Attribute (no parentheses `()`)
# * **Purpose:** Returns an `Index` object containing all column names.
# * **Interpretation:** Use this to verify exact column naming conventions, capitalization, and whitespace.
# 
# ## 3. `df.head(n=5)` — Preview Initial Records
# * **Type:** Method
# * **Purpose:** Returns the first `n` rows of the DataFrame (default is `5`).
# * **Interpretation:** Provides a quick visual inspection of the actual data values and format.
# 
# ## 4. `df.info()` — Data Types & Missing Values
# * **Type:** Method
# * **Purpose:** Prints a concise summary including total entries, column data types (`dtypes`), and non-null (non-missing) counts.
# * **Interpretation:** Essential for spotting **missing values** and detecting incorrect data types (e.g., numbers stored as strings).

# ---
# 
# ### 📋 Quick Reference Summary
# 
# | Command | Type | Purpose | Primary Output |
# | :--- | :--- | :--- | :--- |
# | `df.shape` | Attribute | Check size | `(rows, columns)` tuple |
# | `df.columns` | Attribute | View feature names | Array of column labels |
# | `df.head(n)` | Method | Inspect data sample | First `n` rows (default: 5) |
# | `df.info()` | Method | Check dtypes & nulls | Summary report |

# # Data Se Insights Nikalna (Exploratory Data Analysis)
# 
# 
#  Model banane se pehle data ko samajhna sabse zaroori hai. Insights nikalne ka matlab hai: "Data ke andar chupe hue aise patterns dhoondna jo ghar ki price (SalePrice) ko affect karte hain."
# 
#  Summary:
# 
#  - 1. Target Variable (SalePrice) ka overview: Average price kya hai? Min/Max kya hai?
#  - 2. Numerical Correlation: Kaunse numerical columns (jaise area, quality) price ke sath sabse zyada related hain?
#  - 3. Categorical Impact: Kaunse areas (Neighborhood) ya ghar ki styles (HouseStyle) mehngi hain?

# ## Know your Data:
# 
# ### Will Use this feature while converting into numerical format / Encoding 
# 1. Nighborhooh
# 2. Overall Qual
# 3. YearBuilt
# 4. Foundation
# 5. Electircal
# 6. KitchenQual
# 7. GrageType
# 8. GarageFinish
# 9. Fence

# 

# ## Data-Integration

# In[6]:


df = pd.concat([df_train, df_test])

print(f"Shape of integrated dataframe: {df.shape}")


# In[7]:


df.head(5)


# In[8]:


df.tail() # see the last line of the combined dataframe


# ## Get the brief information of data-set

# In[9]:


df.info()

## Most Null Value for drop

1- Alley
2- FireplaceQu
3- PoolQC
4- Fence
5- MiscFeature

# In[10]:


# Select only the columns that have 'int64' (integer) data type

int_feature = df.select_dtypes(include=['int64']).columns

# Print the total count of these integer columns
print(f"The total number of integer column is: {int_feature.shape[0]}")

#Print the actual names of these columns to check for traps!
print("\nList of integer columns:")
print(int_feature.tolist())  # .tolist() makes it easier to read


# In[11]:


# Select only the columns that have 'float64' (floating point) data type

float_feature = df.select_dtypes(include=['float64']).columns

# Print the total count of these float columns
print(f"The total number of float column is: {float_feature.shape[0]}")

#Print the actual names of these columns to check for traps!
print("\nList of float columns:")
print(float_feature.tolist())  # .tolist() makes it easier to read


# In[12]:


# Select only the columns that have 'object' (string) data type

object_feature = df.select_dtypes(include=['object', 'string']).columns

# Print the total count of these object columns
print(f"The total number of categorical column is: {object_feature.shape[0]}")

#Print the actual names of these columns to check for traps!
print("\nList of categorical columns:")
print(object_feature.tolist())  # .tolist() makes it easier to read


# ## Get the Statistical Information of Numerical Features
Jab tum df.describe() run karoge, toh yeh columns aayenge. Inka matlab samajhna bohat zaroori hai:
1. count: Batata hai ke column mein kitni valid (non-null) values hain. (Missing values check karne ka quick tarika).
2. mean: Average (ausat) value. (Jaise ghar ki average price).
3. std: Standard Deviation. Batata hai ke data average se kitna door/phaila hua hai.
4. min: Sabse chhoti value.
5. 25%: 25 percentile (Q1).
6. 50%: 50 percentile (Q2) ya median.
7. 75%: 75 percentile (Q3).
8. max: Sabse badi value.
# In[13]:


df.describe()

# distribution of numerical features

raw: [1, 3, 4, 5, 2, 7, 6, 8, 0, 9]
             25%           50%     75%
sort: [0, 1, 2.5, 3, 4, | 5, 6, 7, 8, 9]

distribution in %

1 % min = 0
25% =  2.5
50% =  4.5
75% =  7.5
100% max =  9
# In[14]:


df.describe().shape


# ## Data Cleaning
Data cleaning is the process of detecting incomplete,
incorrect, inaccurate or irrelevant parts of the data and
then replacing, modifying or deleting the dirty data.Short Summary:
Whenever there's a missing value (NaN/NA) in the data, a Data Scientist has the option to choose one of these 6 methods.

1. Ignore / Delete the Row (just remove the row itself)
   - Meaning: If there's a missing value in a row (like for a house), then remove that entire row from the dataset.
   - When to use it? When missing values are very few (like out of 1000 houses, only 1 or 2 have missing data).
   - Example: If in a list of 1000 students, only 1 student's "Date of Birth" is missing, then just delete that 1 student's entire line (row).

2. Global Constant / Manual Fill (just write "Unknown" or "0" yourself)
   - Meaning: Wherever the data is missing, just write a fixed value there.
   - When to use it? When a missing value actually means "nothing" or "not applicable."
   - Example: If "GarageType" is missing, then write "No Garage" there.
   - If "PoolArea" is missing, then write 0 there. (We actually did this in the previous steps!)

3. Measure of Central Tendency (Mean, Median, Mode) - ⭐ Most Important
   - When we need to fill in a number or text, we look at the "pattern" of the data:
   - Mean (Average): Add up all the numbers and divide.
    (If there are outliers, DON'T use this, remember the Ferrari example? The mean gets messed up!) 
     Median (50% / The middle value): The exact middle number. (If the data has outliers,
     always use the Median. This is the industry favorite!)

4. Mode (The value that appears most often): The value that repeats the most times.
   (This is used to fill in text/categories like "Neighborhood").  

🚀 Advanced Methods (These Are Used Less in the Industry)

Meaning: First divide the data into groups, then fill it in.

- Example: If "Salary" is missing, don't take the average of the whole company. First take the average of "Managers" and
 fill in the missing salary for a Manager, and take the average of "Clerks" and fill in the missing salary for a Clerk.

 1. Most Probable Value (ML Algorithms)
   - Meaning: Use an AI / Machine Learning model to "guess" (predict) the missing value.
   - Why is it used less? It's very slow and complex. As long as simple Median/Mode is working, nobody uses AI.
# ## Handeling Missing Value

# ## Visualize null/missing-value
Here are those two lines about Data Visualization — the most powerful and commonly used tool in the world of graphs!
Let's understand their working and purpose in very simple words and in the form of comments.


# In[15]:


# 1. Canvas (Drawing Board) ka size set karna
plt.figure(figsize=(16, 9))  
# Purpose: Yeh line ek khali drawing board banati hai jo 16 inch chouda (width) aur 9 inch lamba (height) hai. 
# Isse heatmap bada aur saaf dikhta hai, chota aur crowded nahi hota.

# 2. ka purpose hai: Missing values (NaN/NA) ko ek colorful map ki tarah dekhna.
#Yeh line aapko ek jhalak (snapshot) mein dikha deti hai ke aapke pure dataset mein kahan-kahan data missing hai.
sns.heatmap(df.isnull())

# 3. ka purpose hai: Aapke banaye hue graph (chart) ko ek Image file (jaise .png ya .jpg) ki tarah aapke computer mein save karna.
#Jab aap plt.show() chalate hain, toh graph sirf screen par dikhta hai. Lekin jab aapko yeh graph kisi report,
# sentation, ya apne teacher/boss ko WhatsApp/Email par bhejna ho, toh aap plt.savefig() use karte hain.

plt.savefig("EDA_img/heatmap_DF_of_null_values.png")


# ## Get null value Percentage for every percentage

# In[16]:


# set index as a id column
df.set_index("Id")


# In[17]:


# Calculate missing values for all columns
null_percentage = df.isnull().sum()

# Filter and show ONLY columns that have missing values (> 0)
null_count = null_percentage[null_percentage > 0]

null_count


# In[18]:


# Tell Pandas to show ALL rows, no matter how many there are
pd.set_option('display.max_rows', None)

# Now calculate and print the percentage
null_percentage = (df.isnull().sum()/df.shape[0]*100)


null_percentage

Data will completly remove from the data_set which cantain strong null_percentage

1. Alley
2. PoolQC
3. Fence
4. MiscFeature
5. FireplaceQu
# # Drop Column/Features
# ## As per my observation i will not drop any feature

# In[19]:


# find the columns where missing value percentage is greater than 50%
miss_value_50_perc = null_percentage[null_percentage > 50]

miss_value_50_perc


# In[20]:


"""
As Per our Domain Knowledge, we can not drop columns with more than 50% missing values, instead we should use constant values to fill them 'NA'.
"""

miss_value_50_perc = null_percentage[null_percentage > 50]

miss_value_50_perc


# In[21]:


df["Alley"].value_counts()




# In[22]:


df["PoolQC"].value_counts()


# In[23]:


df["Fence"].value_counts()


# In[24]:


df["MiscFeature"].value_counts()


# In[25]:


"""As per our domain knowledge we will not drop 'FireplaceQu' feature instead None Value add constant Value 'N/A' """

miss_value_20_50_perc = null_percentage[(null_percentage > 20) & (null_percentage <= 50)]
miss_value_20_50_perc


# In[26]:


miss_value_5_20_perc = null_percentage[(null_percentage > 5) & (null_percentage <= 20)]
miss_value_5_20_perc


# In[27]:


df["LotFrontage"].value_counts().head() # importane feature we will not remove becuase it tells us street distance


# In[28]:


# 1. Get the names of columns that have 5% to 20% missing values
# 2. Select only those specific columns from the main dataframe 'df'
# 3. Check for missing values in these selected columns (returns True/False)
# 4. Draw a heatmap to visually see where the data is missing (Yellow = Missing, Dark = 

sns.heatmap(df[miss_value_5_20_perc.keys()].isnull())


# ## Mising Value Imputation:
# 
#  - Missing Value Imputation means filling in the empty (missing/NaN) values that exist in the data..
#  - Machine Learning models can't understand empty data (NaN) and throw an error,
#  - which is why we replace these missing values with some logical value.

# ## Key Points (80/20 Rule for Industry)
# 
# In the industry, we don't use super complex imputation every day. In 80% of cases, these simple techniques get the job done:
# 
# 1. Mean (Average): Use it when all numbers are normal and balanced — no crazy big or crazy small values.
# 2. Median (Middle Value): Use it when some numbers are crazy big or crazy small (these are called outliers).
# 3. Why Median? Because Median doesn't care about crazy big or small numbers. It stays safe in the middle.
# 
# For Text / Categories (Categorical Data):
# 1. Mode (Most Frequent): Fill the missing spot with whatever value shows up the most. Simple.
# 
# Golden Rule of ML (Very Important!):
# 1. First, teach the imputer using only the Training data (fit).
# 2. Then, use that same learning to fill missing values in both Training and Test data (transform).
# 3. Why? So that "Data Leakage" doesn't happen. (Data Leakage = when test data info sneaks into training and cheats the results.)
# 

# In[29]:


# get the total_missing value feature

missing_value_feat = null_percentage[null_percentage > 0]

print(f"Total missing value features: {len(missing_value_feat)}")


# In[30]:


missing_value_feat


# In[31]:


# find text columns with missing data to apply mode imputation later
cat_na_feat = missing_value_feat[missing_value_feat.keys().isin(object_feature)]

print(f"Total number of categorical missing features: {len(cat_na_feat)} ")

cat_na_feat


# In[32]:


# filter integer columns that have missing values
int_na_feat = missing_value_feat[missing_value_feat.keys().isin(int_feature)]

print(f"Total number of integer missing features: {len(int_na_feat)} ")

int_na_feat


# In[33]:


# filter float columns that have missing values
float_na_feat = missing_value_feat[missing_value_feat.keys().isin(float_feature)]

print(f"Total number of float missing features: {len(float_na_feat)} ")

float_na_feat


# ### Handling MSZoning -->  0.137033

# In[34]:


df["MSZoning"].value_counts()


# In[35]:


sns.countplot(df["MSZoning"])


# ### Backup of original data

# In[36]:


df_mvi = df.copy()
df_mvi.shape


# In[37]:


# find the most frequent value in the column
mszoning_mode = df["MSZoning"].mode()[0]

# replace NaN safely without using inplace=True
df_mvi["MSZoning"] = df_mvi["MSZoning"].replace(np.nan, mszoning_mode)

# verify that no missing values are left (should output 0)
df_mvi["MSZoning"].isnull().sum()


# In[38]:


# plot vertical count of each category in MSZoning
sns.countplot(x="MSZoning", data=df_mvi)


# In[39]:


def oldNewCountPlot(df, df_new, feature):
    # reset index inside function to prevent duplicate label errors
    df = df.reset_index(drop=True)
    df_new = df_new.reset_index(drop=True)

    # create left subplot for original data
    plt.subplot(121)
    sns.countplot(x=feature, data=df)
    plt.title("Old Data Distribution")

    # create right subplot for new data
    plt.subplot(122)
    sns.countplot(x=feature, data=df_new)
    plt.title("New Data Distribution")

    # display both plots side by side
    plt.show()

# call the function
oldNewCountPlot(df, df_mvi, "MSZoning")


# ### Handing Alley = 93.216855

# In[40]:


df_mvi["Alley"].value_counts()


# 

# In[41]:


# fill missing values with "NA" (no alley access)
alley_constant = "NA"

# fill missing values in the "Alley" column with the defined constant
df_mvi["Alley"] = df_mvi["Alley"].fillna(alley_constant)

# verify and display result

missing_count = df_mvi["Alley"].isnull().sum()
print(f"Missing Values: {missing_count}")
print("\nUpdated distribution:")
print(df_mvi["Alley"].value_counts())


# In[42]:


# visualize the updated Alley distribution
sns.countplot(x="Alley", data=df_mvi)
plt.title("Alley Distribution After Imputation")
plt.show()


# In[43]:


# define a resuable function that takes 3 inputs : original data , new_data, and column

def oldNewCountPlot(df, df_new, feature):
    """
    Compare distribution of a new feature before and after imputation.
    create side-by-side vertical countplots for visual verification 
    """

    # reset index of original data to remove duplicate labels (prevents seaborn error)
    df = df.reset_index(drop=True)


    #create the first subplot on the left side (1 raw, 2 columns , position 1)
    plt.subplot(121)

    # plot vertical bars of the feature from the original data
    sns.countplot(x=feature, data=df)
    plt.title("Old Data Distribution")

    # create the second subplot on the right side (1 row, 2 columns, position 2)
    plt.subplot(122)

    # plot vertical bars of the feature from the new data
    sns.countplot(x=feature, data=df_new)
    plt.title("New Data Distribution")

    # display the plots side by side
    plt.show()
oldNewCountPlot(df, df_mvi, "Alley")


# In[44]:


print("Data Type of LotFrontage")
print(df["LotFrontage"].dtype)

print(df["LotFrontage"].head())


# ## Handling Lotfrontage

# LotFrontage = 16.649538
📊 Part 1: Mean vs Median - Kab Kya Use Karein?

🎯 Simple Rule (Yaad Rakhein)

Mean (Average) Use Karein Jab:

* Data normally distributed ho (bell curve shape)
* Outliers na hon (bohat bari ya choti values na hon)
* Example: Heights, weights, test scores

Median (Middle Value) Use Karein Jab:
* Data mein outliers hon (extreme values)
* Data skewed ho (ek side zyada)
* Example: Income, house prices, LotFrontage💡 Visualization Se Kaise Pata Karein?

* Agar box balanced hai (median line beech mein) → Mean use karo
* Agar box skewed hai (median line ek side) ya outliers hain → Median use karo


Histogram/Distribution Dekhein:

* Agar bell curve shape hai (symmetric) → Mean use karo
* Agar ek side zyada hai (skewed) → Median use karo
# In[45]:


# define function to visualize numerical data distribution
# take data-frame (or series) and optional figure size as input
def boxHisplot(df, figsize=(16, 5)):
    """
    Create side-by-side boxplot & histogram for numerical data analysis.
    Helps decide whether to use mean or median for numerical data.
    """

    # reset the index to remove duplicate labels (this fixes the seaborn error!)
    df = df.reset_index(drop=True)

    # create a figure with specified size (default 16x5 inches)
    plt.figure(figsize=figsize)

    # create left subplot for boxplot (1 row, 2 columns, position 1)
    plt.subplot(1, 2, 1)

    # draw VERTICAL boxplot using x= parameter 
    # x= makes the boxplot stand upright instead of lying down
    sns.boxplot(x=df)

    # add a title to identify the plot
    plt.title("Boxplot - Check for Outliers")

    # create right subplot for histogram (1 row, 2 columns, position 2)
    plt.subplot(1, 2, 2)

    # draw histogram to see data distribution shape
    # histogram shows frequency of values in bins, kde adds a smooth curve
    sns.histplot(data=df, bins=30, kde=True)

    # add a title to identify the plot
    plt.title("Histogram - Check for Skewness")

    # adjust layout so titles and labels don't overlap
    plt.tight_layout()

    # display the plots on the screen
    plt.show()

# analyze LotFrontage column to decide mean or median
boxHisplot(df["LotFrontage"])


# In[46]:


lot_frontage_mean = df["LotFrontage"].mean()
lot_frontage_mean


# In[47]:


# step 1: calculate the mean from the original dataframe (prevents data leakage)
lot_frontage_mean = df["LotFrontage"].mean()

# step 2: fill missing values in df_mvi safely using direct assignment
# (avoiding inplace=True to prevent Pandas 2.0 errors)
df_mvi["LotFrontage"] = df_mvi["LotFrontage"].fillna(lot_frontage_mean)

# step 3: verify that no missing values are left in the column
print("Missing values after imputation:", df_mvi["LotFrontage"].isnull().sum())


# In[48]:


# recreate df_mvi as a full DataFrame copy to fix the Series error
df_mvi = df.copy()

# now check if the column exists (this will work perfectly now)
print("LotFrontage in df_mvi:", "LotFrontage" in df_mvi.columns)


# In[49]:


# create a copy of orginial data_frmae

df_mvi = df.copy()

# confirm is the shape is same or not

print(df_mvi.shape)


# ## MSZoning Mode Imputation
# 

# In[50]:


msznoning_mode = df["MSZoning"].mode()[0]
print(f"MSZonining mode: {msznoning_mode}")

# Fill the missing values to mode
df_mvi["MSZoning"] = df_mvi["MSZoning"].replace(np.nan, msznoning_mode)

# verification 
print(f"MSZoning missing after imputation: {df_mvi['MSZoning'].isnull().sum()}")


# ## Alley missing values mean "No access"

# In[51]:


df_mvi["Alley"] = df_mvi["Alley"].fillna("NA")

# After verification
print(f"Alley missing after imputation: {df_mvi['Alley'].isnull().sum()}")
print("\nAlley Counts: ")
print(df_mvi["Alley"].value_counts())


# # Calculate Mean of LotFrontage

# In[52]:


# calculate the mean of lot frontage

lot_frontage_mean = df["LotFrontage"].mean()
print(f"LotFrontage mean: {lot_frontage_mean}")

# Fill the missing values with the mean
df_mvi["LotFrontage"] = df_mvi["LotFrontage"].replace(np.nan, lot_frontage_mean)

# verification after imputation
print(f"LotFrontage missing after imputation: {df_mvi['LotFrontage'].isnull().sum()}")


# In[53]:


# Define a function to compare old and new distributions for a numerical feature
def oldNewBoxHistPlot(df, df_new, feature, figsize=(16, 10)):
    """
    Compare old and new distributions of a numerical feature.
    Creates 4 plots: old boxplot, old histogram, new boxplot, new histogram.
    """

    # Reset the index of the old dataframe to avoid duplicate label errors
    df = df.reset_index(drop=True)

    # Reset the index of the new dataframe to avoid duplicate label errors
    df_new = df_new.reset_index(drop=True)

    # Create a blank canvas with the given figure size
    plt.figure(figsize=figsize)

    # --- Plot 1: Old Data Boxplot (top-left position) ---
    # Create the first subplot in a 2x2 grid at position 1
    plt.subplot(2, 2, 1)
    # Draw a boxplot for the old data to see outliers and spread
    sns.boxplot(x=df[feature])
    # Add a title to the first plot
    plt.title("Old Data - Boxplot")

    # --- Plot 2: Old Data Histogram (top-right position) ---
    # Create the second subplot in a 2x2 grid at position 2
    plt.subplot(2, 2, 2)
    # Draw a histogram with a smooth curve (KDE) to see the distribution shape
    sns.histplot(df[feature], kde=True)
    # Add a title to the second plot
    plt.title("Old Data - Histogram")

    # --- Plot 3: New Data Boxplot (bottom-left position) ---
    # Create the third subplot in a 2x2 grid at position 3
    plt.subplot(2, 2, 3)
    # Draw a boxplot for the new data to see outliers and spread after imputation
    sns.boxplot(x=df_new[feature])
    # Add a title to the third plot
    plt.title("New Data - Boxplot")

    # --- Plot 4: New Data Histogram (bottom-right position) ---
    # Create the fourth subplot in a 2x2 grid at position 4
    plt.subplot(2, 2, 4)
    # Draw a histogram with a smooth curve (KDE) for the new data
    sns.histplot(df_new[feature], kde=True)
    # Add a title to the fourth plot
    plt.title("New Data - Histogram")

    # Adjust the spacing between plots so they do not overlap
    plt.tight_layout()

    # Display all four plots on the screen
    plt.show()

 # Call the function to compare old and new distributions of LotFrontage
oldNewBoxHistPlot(df, df_mvi, "LotFrontage")   


# ### Handling Utilities = 0.068517

# 

# In[54]:


# show the count of each category in the 'Utilities' column
df['Utilities'].value_counts()


# In[55]:


# Find the most frequent values in the 'Utilities' column
utilities_Mode = df['Utilities'].mode()[0]

# Fill missing values in df_mvi with the mode value 
df_mvi['Utilities'] = df_mvi['Utilities'].replace(np.nan,utilities_Mode)


# Verify that no missing values are left
df_mvi['Utilities'].isnull().sum()


# ### Handling Exterior1st = 0.034258 and Exterior2nd = 0.034258

# In[56]:


df["Exterior1st"].value_counts()


# In[57]:


df["Exterior2nd"].value_counts()


# In[58]:


# Find the most frequent value in the Exterior1st column
exterior1st_mode = df["Exterior1st"].mode()[0]


# Find the most frequent value in the Exterior2nd column
exterior2nd_mode = df["Exterior2nd"].mode()[0]

# Fill missing values in Exterior1st with its mode value
df_mvi["Exterior1st"] = df_mvi["Exterior1st"].replace(np.nan, exterior1st_mode)

# Fill missing values in Exterior2nd with its mode value
df_mvi["Exterior2nd"] = df_mvi["Exterior2nd"].replace(np.nan, exterior2nd_mode)

# Verify that no missing values are left in Exterior1st
print("Exterior1st null count:", df_mvi["Exterior1st"].isnull().sum())

print("Exterior2nd null count:", df_mvi["Exterior2nd"].isnull().sum())


# ### Handling MasVnrType = 60.500171 and MasVnrArea = 0.787941

# In[59]:


#create a heatmap to visualize missing values in both columns

sns.heatmap(df[["MasVnrType", "MasVnrArea"]].isnull())

plt.title("Missing Values: MasVnrType and MasVnrArea")
plt.show()


# In[60]:


df[df[["MasVnrType", "MasVnrArea"]].isnull().any(axis=1)]


# In[61]:


df["MasVnrType"].value_counts()


# In[62]:


# Find the most frequent value in the MasVnrType column
mass_vnr_type_mode = df["MasVnrType"].mode()[0]

# Print the mode value to check it
print(f"MasVnrType mode: {mass_vnr_type_mode}")


# Fill the missing values in the MasVnrType column with the mode value
df_mvi["MasVnrType"] = df_mvi["MasVnrType"].replace(np.nan, mass_vnr_type_mode)

# Verify that no missing values are left
print(f"MasVnrType missing values after imputation: {df_mvi['MasVnrType'].isnull().sum()}")


# In[63]:


# Define a function to show boxplot and histogram for a single feature
def boxHistPlot(series, figsize=(12, 5)):
    """
    Show boxplot and histogram for a single numerical feature.
    """

    # Reset the index of the series to avoid duplicate label errors
    series = series.reset_index(drop=True)

    # Create a blank canvas with the given figure size
    plt.figure(figsize=figsize)

    # --- Plot 1: Boxplot (left side) ---
    # Create the first subplot in a 1x2 grid at position 1
    plt.subplot(1, 2, 1)
    # Draw a boxplot to see outliers and spread
    sns.boxplot(x=series)
    # Add a title
    plt.title("Boxplot")

    # --- Plot 2: Histogram (right side) ---
    # Create the second subplot in a 1x2 grid at position 2
    plt.subplot(1, 2, 2)
    # Draw a histogram with a smooth curve (KDE)
    sns.histplot(series, kde=True)
    # Add a title
    plt.title("Histogram")

    # Adjust spacing so plots do not overlap
    plt.tight_layout()

    # Display both plots
    plt.show()


boxHistPlot(df["MasVnrArea"])
boxHistPlot(df_mvi["MasVnrArea"])       


# In[64]:


# Set the mode value for MasVnrArea as 0

masvnrmode_area = 0

# Fill missing values in df_mvi with the mode value (0)
df_mvi["MasVnrArea"] = df_mvi["MasVnrArea"].replace(np.nan, masvnrmode_area)

# Verify that no missing values are left
print("MasVnrArea missing after imputation:", df_mvi["MasVnrArea"].isnull().sum())


# ### Handling BSmt Handling Feature

cat_bsmt_feat = 
BsmtQual        2.774923
BsmtCond        2.809181
BsmtExposure    2.809181
BsmtFinType1    2.706406
BsmtFinType2    2.740665

num_bsmt_feat = 
BsmtFinSF1      0.034258
BsmtFinSF2      0.034258
BsmtUnfSF       0.034258
TotalBsmtSF     0.034258
BsmtFullBath    0.068517
BsmtHalfBath    0.068517
# In[65]:


# Create a list of Categorical basement features
cat_bsmt_feat = [
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",

]

# print the list to verify
print("Categorical basement features:")
print(cat_bsmt_feat)


# In[66]:


# Create a list of Numerical basement features
num_bsmt_feat = [
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath"
]

# print the list to verify
print("Numerical basement features:")
print(num_bsmt_feat)


# In[67]:


# Understand the Pattern via heatmap
sns.heatmap(df[cat_bsmt_feat + num_bsmt_feat].isnull())
plt.title("Missing Values Pattern: Basement Features")
plt.show()


# In[69]:


# Loop through each categorical basement feature
for feature in cat_bsmt_feat:
    print(f"value count of {feature}: {df[feature].value_counts()}")


# In[70]:


# Loop through each categorical basement feature
for feature in cat_bsmt_feat:
    # Fill the missing value with "NA" (No Basement)
    df_mvi[feature] = df_mvi[feature].fillna("NA")

    # Print the feature name and missing count after imputation
    print(f"{feature} missing after imputation : {df_mvi[feature].isnull().sum()}")


# In[73]:


df_bsmt = df[cat_bsmt_feat + num_bsmt_feat]
df_bsmt[df_bsmt.isnull().any(axis=1)]


# In[74]:


# Loop through each numerical basement feature
for feature in num_bsmt_feat:
    #Fill missing values with 0 (No Basement)
    df_mvi[feature] = df_mvi[feature].fillna(0)

    #print the feature name and missing count after imputation
    print(f"{feature} missing count after imputation: {df_mvi[feature].isnull().sum()}")


# ### Handling Electrical = 0.034258 KitchenQual = 0.034258

# In[76]:


df["Electrical"].value_counts()


# In[77]:


df["KitchenQual"].value_counts()


# In[78]:


# Show rows where Electrical, KitchenQual, or KitchenAbvGr is missing
df_ekk = df[[ "Electrical", "KitchenQual", "KitchenAbvGr"]]
df_ekk[df_ekk.isnull().any(axis=1)]


# In[79]:


# Find the most frequent value in the Electrical column
electrical_mode = df["Electrical"].mode()[0]


# Fill missing values in the df_mvi with the mode value
df_mvi["Electrical"] = df_mvi["Electrical"].fillna(electrical_mode)


# Verify that no missing values are left
print("Electrical missing after imputation: ", df_mvi["Electrical"].isnull().sum())


# ### Handling Remaining Cat Feature
Functional = 0.068517  --> mode
FireplaceQu = 48.646797 --> NA
PoolQc = 99.657417 --> NA
Fence = 80.438506 --> NA
MiscFeature = 96.402878 --> NA
SaleType = 0.034258  --> mode
# In[81]:


df["Functional"].value_counts()


# In[82]:


df["SaleType"].value_counts()


# In[83]:


# Find the most frequent value in the Functional column

functional_mode = df["Functional"].mode()[0]

# Print the mode value to check it
print(f"Functional Mode: {functional_mode}")


# Fill missing values in df_mvi with the mode value
df_mvi["Functional"] = df_mvi["Functional"].replace(np.nan, functional_mode)

# Verify that no missing values are left
print("Functional missing after imputation: ", df_mvi["Functional"].isnull().sum())


# In[84]:


# Find the most frequent value in the SaleType column

saletype_mode = df["SaleType"].mode()[0]

# Print the mode value to check it
print(f"SaleType Mode: {saletype_mode}")

# Fill missing values in df_mvi with the mode value
df_mvi["SaleType"] = df_mvi["SaleType"].replace(np.nan, saletype_mode)

# Verify that no missing values are left
print("SaleType missing after imputation:", df_mvi["SaleType"].isnull().sum())


# In[85]:


# Create a list of other categorical features with high missing values

other_cat_feat = ["FireplaceQu", "PoolQC", "Fence", "MiscFeature"]

# Loop through each feature and print its value counts
for feat in other_cat_feat:
    print(f"Value count of {feat}:")
    print(df[feat].value_counts())
    print("-" * 40)


# In[86]:


# Fill missing values in FireplaceQu with "NA" (No Fireplace)
df_mvi["FireplaceQu"] = df_mvi["FireplaceQu"].fillna("NA")
print("FireplaceQu missing after imputation:", df_mvi["FireplaceQu"].isnull().sum())

# Fill missing values in PoolQC with "NA" (No Pool)
df_mvi["PoolQC"] = df_mvi["PoolQC"].fillna("NA")
print("PoolQC missing after imputation:", df_mvi["PoolQC"].isnull().sum())

# Fill missing values in Fence with "NA" (No Fence)
df_mvi["Fence"] = df_mvi["Fence"].fillna("NA")
print("Fence missing after imputation:", df_mvi["Fence"].isnull().sum())

# Fill missing values in MiscFeature with "NA" (None)
df_mvi["MiscFeature"] = df_mvi["MiscFeature"].fillna("NA")
print("MiscFeature missing after imputation:", df_mvi["MiscFeature"].isnull().sum())

