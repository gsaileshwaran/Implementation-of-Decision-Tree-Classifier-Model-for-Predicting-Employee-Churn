# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the dataset and required libraries (pandas, sklearn, matplotlib).
2. Encode the categorical column (salary) using LabelEncoder.
3. Split the dataset into features (x) and target (y), then apply train_test_split.
4. Train a DecisionTreeClassifier using the training data and make predictions.
5. Evaluate the model's accuracy and visualize the decision tree using plot_tree.
6. 
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Saileshwaran Ganesan
RegisterNumber:  212224230237
*/
```

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
```
```
data = pd.read_csv("Employee.csv")
```
```
print(data.head())
```
```
print(data.info())
print(data.isnull().sum())
```
```
print(data["left"].value_counts())
```
```
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
print(data.head())
```
```
x = data[["satisfaction_level", "last_evaluation", "number_project", 
          "average_montly_hours", "time_spend_company", 
          "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]
```
```
y_pred = dt.predict(x_test)
```
```
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)
```
```
sample_prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Predicted result for new employee data:", sample_prediction)
```
```
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```






## Output:
Displaying first few rows

![image](https://github.com/user-attachments/assets/abbcda4e-8979-431b-9aa3-3129980206f4)
Checking data information and null values

![image](https://github.com/user-attachments/assets/b80f43e1-6a17-4044-92b6-a42fe5f2edf4)
Viewing the distribution of target variable

![image](https://github.com/user-attachments/assets/bb0cf16f-35c8-428d-956a-344367b9c71e)
Encoding the 'salary' column

![image](https://github.com/user-attachments/assets/38264c5c-3e88-4034-a2e8-9829c7f1a9cd)
Training the Decision Tree model

![image](https://github.com/user-attachments/assets/892287e3-4048-487e-8c3c-ec575c11a3ee)
Calculating accuracy

![image](https://github.com/user-attachments/assets/5428b7fc-0520-4b5f-979a-57d166a672c5)
Predicting for a new sample input

![image](https://github.com/user-attachments/assets/45bf2e8c-54a1-4ef9-a842-3e4f8057b762)
Visualisation

![image](https://github.com/user-attachments/assets/4908248b-a6b5-42f9-968a-2011bd24d91a)











## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
