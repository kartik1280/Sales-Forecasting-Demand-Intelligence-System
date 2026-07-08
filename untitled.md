# TASK 1
first things first we load the dataset 

then we do df.head() to see first 5 rows 
then we do df.shape to see its dimensions (rows x columns) -> 9800x18
then we do df.columns to see the different columns 
then we do df.info()

from df.info() we saw that order and ship date both were object datatypes and not dates 
and Postal Code has 9789 non-null values when there are total 9800 rows so postal code column has some missing values (11)

first lets tackle with the date problem 

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

i ran these two but got an error 
python's default date format is MM/DD/YYYY 
and the dataset had this format DD/MM/YYYY

so now we have to run this instead 
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

the missing values of postal code -> lets really think about it do we need it to predict anything? 
the answer is no we don't need it so there is no need to fix this problem 
if we drop them then we will be losing valid sales information 

df.duplicated().sum()
there are no duplicate values

ok we extracted day month year week and season in seperate columns this will help us make forcasting models 

now we will aggregate daily sale 
aggregate basically means - Combine many rows into a summary
Instead of three rows for 1 Jan with seperate data, we want 1 row with aggregate data 

| Library         | Mainly Used For                    |
| --------------- | ---------------------------------- |
| Pandas          | Data manipulation                  |
| NumPy           | Mathematical operations            |
| Matplotlib      | Visualization                      |
| Scikit-learn    | Machine Learning                   |
| **Statsmodels** | Statistical & Time Series Analysis |


### business question 1 
Which product category generates the highest total revenue?

so we ran 
df.groupby("Category")["Sales"].sum() 

this basically means to group category on the basis of sum of sales 
this will tell us which category had the highest sales (highest total revenue)

why do we need to know which category generates the highest toal revenue?
lets say technology makes more revenue (8 crores)
and furniture makes less (5 crores) 

the management needs to know this for 
inventory - should they store more chairs or more computers 
marketing - where should their advertisement budget go
expansion - which product line should get more investment 

### business question 2 
Which region has the most consistent sales growth over 4 years?

for this we ran 
region_year_sales = (
    df.groupby(["Year", "Region"])["Sales"]
      .sum()
      .reset_index()
)

group region and year and calculate the sum of sales for each group 

total 4 regions - east west central south 
with this information we plot a graph and see which region has the most CONSISTENT growth and that is going to be the east 

why do we wanna know this? 
because stable and consistent growth is predictible 
which helps in forecasting 

### Business Question 3 
What is the average time between Order Date and Ship Date?

for this we simply did (ship date - order date)
df["Shipping_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Shipping_Time"].mean()

the average time was 3.9611224489795918 days which is approx 4 days 
then we also grouped it by regions 
to see if it varies 

East       3.910233
West       3.930255
South      3.961202
Central    4.065876

Among all regions:
- East has the fastest average shipping time (3.91 days).
- West and South have similar shipping times.
- Central has the slowest average shipping time (4.07 days).
but the difference is very small indicating that shipping is consistent accross all regions

### Business Question 4 
Are there months that consistently spike across all years?

for this we grouped sales by year and month 
	Year	Month	Sales
0	2015	1	14205.707
1	2015	2	4519.892
2	2015	3	55205.797
3	2015	4	27906.855
4	2015	5	23644.303

then we plotted a graph and we saw how some months have a spike 
The monthly sales trends indicate clear seasonality. Sales consistently increase during the final months of the year, particularly in November, which shows the highest sales across all four years. December also maintains relatively strong sales levels, while January and February generally record lower sales. This pattern suggests increased customer purchasing during the holiday season, making seasonality an important factor for future sales forecasting.


# TASK 2

what is time series analysis 

1. Collect Data
        ↓
2. Clean & Prepare Data
        ↓
3. Convert to Time Series
        ↓
4. Visualize the Time Series
        ↓
5. Decompose the Series
        ↓
6. Check Stationarity
        ↓
7. Make Stationary (Differencing if needed)
        ↓
8. Build Forecasting Model
        ↓
9. Evaluate the Model
        ↓
10. Forecast Future Values

we already extracted time features and aggregated them daily/weekly and monthly 
now lets understand what a time series is 

A time series is simply data collected over time in chronological order
Before applying any statistical model, we want to visually inspect the data

Observed Sales
      =
Trend
+
Seasonality
+
Residual

now we will apply time series decomposition 

##### TREND -  the long-term direction ignoring all the little ups and downs 
##### SEASONALITY - A pattern that repeats at regular intervals
##### RESIDUAL (NOISE) - Even after removing trend and seasonality there's still randomness.
##### TIME SERIES - Data collected over time in chronological order
##### STATIONARITY - A time series is called stationary if its statistical properties remain approximately constant over time.

Imagine daily temperature inside an air-conditioned office 
24
25
24
26
25
24
25
Every day it fluctuates slightly around the same average.
Nothing changes dramatically.
This is stationary.

if the average itself is increasing overtime its not stationary 

#### Why do models like SARIMA need stationarity?

Think about learning to drive.
Suppose every day:
Road rules stay the same.
Easy to learn.
Now imagine:
Every hour
traffic rules change
speed limits change
signals change
You'd struggle to learn a consistent pattern.
SARIMA works the same way.
It assumes that the underlying behavior of the data stays reasonably stable over time. If the average and trend keep changing, the model has difficulty separating the long-term growth from the repeating patterns.

That's why we first test for stationarity and, if necessary, transform the data before fitting the model.

TO CHECK STATIONARITY we have Next Step: "Augmented Dickey-Fuller" (ADF) Test
Now we'll let statistics answer the question instead of relying only on the graph.

# TASK 3 

three different ways of forecasting 

###### Expert 1: Statistician (SARIMA)
Seasonal AutoRegressive Integrated Moving Average
Looks only at historical patterns.

Questions:
What happened last month?
What happens every December?
Is there a trend?
Uses mathematical equations.

###### Expert 2: Business Analyst (Prophet)

Looks for:
Trend
Seasonality
Holiday effects
Designed by Meta specifically for business forecasting.
Very easy to use.

###### Expert 3: Machine Learning Engineer (XGBoost)

Treats forecasting as a regression problem.
Instead of mathematical time-series equations, it learns relationships from features like:
Last month's sales
2 months ago
3 months ago
Rolling average
Quarter
Season

### Understanding SARIMA parameters 

SARIMA(p,d,q)(P,D,Q,m)

(p,d,q) understanding part 
(P,D,Q,m) seasonal part 

###### p is AutoRegression - should i refer to previous data? 
think of it as memory 

if p=1 it will look into one month of previous data 
similarly if p=3 it will look into three 

###### d is Differencing 
only needed if data is not stationary 
but ours is stationary so 
d=0

###### q is Moving Average 
the model considers the previous prediction error.

###### m is How long is one seasonal cycle?

# TASK 5 

What is Anomaly Detection?

An anomaly (also called an outlier) is a data point that is very different from the normal pattern.

For sales forecasting, anomalies could be:

Huge sales spikes
Sudden sales drops
Unexpected events

Method 1
Isolation Forest
Machine Learning approach

Isolation Forest is an unsupervised machine learning algorithm.
Notice:
There are no labels.
We're not telling the model
"This week is abnormal."
Instead it discovers unusual weeks by itself.

Method 2
Z-Score using Rolling Mean
Statistical approach

Z-score measures exactly how many standard deviations away a point is.

Then compare both.

# TASK 6 

K-Means is an unsupervised learning algorithm.
K-Means has no target variable.

It simply receives:
Product Features
↓
Looks for similarity
↓
Creates groups

K = number of clusters
K= 3 
Create 3 groups


Before using K-Means, we need to answer an important question:
How many clusters should we create?

for this we use The Elbow Method 