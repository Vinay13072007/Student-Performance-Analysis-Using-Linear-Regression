import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Food_Delivery_Times.csv")

print("\nShape of Dataset:")
print(data.shape)

print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())


#histplot

plt.figure()
sns.histplot(data['math score'], color='blue', label='Math')
sns.histplot(data['reading score'], color='red', label='Reading')
sns.histplot(data['writing score'], color='green', label='Writing')
plt.legend()
plt.title("Comparison of Subject Score Distributions")
plt.show()


plt.figure()
sns.histplot(data['math score'])
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.histplot(data['reading score'])
plt.title("Distribution of Reading Scores")
plt.xlabel("Reading Score")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.histplot(data['writing score'])
plt.title("Distribution of Writing Scores")
plt.xlabel("Writing Score")
plt.ylabel("Frequency")
plt.show()


# Scatter Plot
plt.figure()
sns.scatterplot(x=data['math score'],y=data['reading score'])
plt.title("Math vs Reading Scores")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.show()

plt.figure()
sns.scatterplot(x=data['reading score'],y=data['writing score'])
plt.title("Reading vs Writing Scores")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.show()

plt.figure()
sns.scatterplot(x=data['math score'],y=data['writing score'])
plt.title("Math vs Writing Scores")
plt.xlabel("Math Score")
plt.ylabel("Writing Score")
plt.show()

#countplot
plt.figure()
sns.countplot(x='gender', data=data)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

#barplot
plt.figure()
sns.barplot(x='gender',y='math score',data=data)
plt.title("Average Math Score by Gender")
plt.show()

plt.figure()
sns.barplot(x='gender',y='reading score',data=data)
plt.title("Average Reading Score by Gender")
plt.show()

plt.figure()
sns.barplot(x='gender',y='writing score',data=data)
plt.title("Average Writing Score by Gender")
plt.show()

#boxplot
plt.figure()
sns.boxplot(x='gender',y='math score',data=data)
plt.title("Math Score Distribution by Gender")
plt.show()

plt.figure()
sns.boxplot(x='gender',y='reading score',data=data)
plt.title("Reading Score Distribution by Gender")
plt.show()

plt.figure()
sns.boxplot(x='gender',y='writing score',data=data)
plt.title("Writing Score Distribution by Gender")
plt.show()

#heatmap
plt.figure()
correlation = data.corr(numeric_only=True)
sns.heatmap(correlation,annot=True)
plt.title("Correlation Heatmap")
plt.show()

#pairplot
sns.pairplot(data)
plt.show()

# Predict Writing Score using Reading Score

X = data[['reading score']]
Y = data['writing score']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)

r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error:", mse)

print("R2 Score:", r2)

plt.figure()
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.title("Linear Regression Line")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.show()

