import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import json

apps = pd.read_csv('./DataFiles/apps.csv')
user_reviews = pd.read_csv('./DataFiles/user_reviews.csv')

# load and create dataset
apps_copy = apps[["App", "Category", "Rating", "Size", "Type", "Price", "Genres"]].copy()
user_reviews_copy = user_reviews[["App", "Sentiment", "Sentiment_Polarity", "Sentiment_Subjectivity"]].copy()
# remove rows without sentiment value
user_reviews_copy = user_reviews_copy.dropna(axis=0, subset=["Sentiment"])
# merge DataFrames
app_review_data = pd.merge(apps_copy, user_reviews_copy, how='inner')
# app_review_data.set_index('App', inplace=True)

# split between “data” and “labels”
label_data = app_review_data["Sentiment"]
app_review_data = app_review_data.drop(["Sentiment"], axis="columns")

# Clean data & missing values

# imputing Sentiment_Polarity, Sentiment_Subjectivity, Rating
new_cols = SimpleImputer(missing_values=np.NaN, strategy='mean')
new_cols = new_cols.fit_transform(app_review_data[['Sentiment_Polarity', 'Sentiment_Subjectivity', 'Rating']])
app_review_data[['Sentiment_Polarity', 'Sentiment_Subjectivity', 'Rating']] = new_cols

# imputing Type, Category, Size, Genres, Price
new_cols = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
new_cols = new_cols.fit_transform(app_review_data[['Type', 'Category', 'Size', 'Genres', 'Price']])
app_review_data[['Type', 'Category', 'Size', 'Genres', 'Price']] = new_cols

new_cols = SimpleImputer(missing_values='Varies with device', strategy='most_frequent')
new_cols = new_cols.fit_transform(app_review_data['Size'].to_numpy().reshape(-1, 1))
app_review_data['Size'] = new_cols

# convert string to numbers
app_review_data['Price'] = app_review_data['Price'].str.replace('$', '')


def convert(x):
    if 'k' in x:
        if '.' in x:
            return x.replace('k', '00').replace('.', '')
        else:
            return x.replace('k', '000')
    else:
        if '.' in x:
            return x.replace('M', '00000').replace('.', '')
        else:
            return x.replace('M', '000000')


app_review_data['Size'] = app_review_data['Size'].apply(lambda x: convert(x))

# One Hot Vector App, Category, Type, Genres
for column in ["App", "Category", "Type", "Genres"]:
    dummies = pd.get_dummies(app_review_data[column])
    app_review_data[dummies.columns] = dummies

app_review_data = app_review_data.drop(["App", "Category", "Type", "Genres"], axis="columns")

# Normalize Size, Rating
max_size = app_review_data["Size"].max()
app_review_data["Size"] = app_review_data["Size"].apply(lambda x: int(x) / int(max_size))

max_rating = app_review_data["Rating"].max()
app_review_data["Rating"] = app_review_data["Rating"].apply(lambda x: float(x) / float(max_rating))

# Split groups to train and test
x = app_review_data
y = label_data

X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)

# Choose k and train KNN model
knn_results = {}
for k in range(1, 21):
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the data
    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)
    # check accuracy of our model on the test data
    knn_results[k] = accuracy_score(y_test, predicted)

with open('knn_result.json', 'w') as file:
    json.dump(knn_results, file)
