import numpy as np
import pandas as pd
data = pd.read_csv("language.csv") 
print(data.head())
print(len(data))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
print(data)
print(data.isnull().sum())
print(data['language'].value_counts())
x = np.array(data['Text'])
y = np.array(data['language'])
print(x)
print(y)

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train)

model = MultinomialNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
user = input("Enter a text: ")
data = cv. transform([user]).toarray()
output = model.predict(data)
print (output)