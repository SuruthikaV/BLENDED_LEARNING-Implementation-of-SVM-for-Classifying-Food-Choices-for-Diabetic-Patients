# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by:SURUTHIKA V 
RegisterNumber: 212225040441 
*/
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('food_items_binary.csv')

print(data.head())
print(data.columns)

features=['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target='class'
x=data[features]
y=data[target]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

scaler=StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

svm=SVC()
param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear', 'rbf'],
    'gamma':['scale','auto']
}

grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(x_train,y_train)

best_model=grid_search.best_estimator_
print("Name: SURUTHIKA V")
print("Register Number: 212225040441")
print("Best Parameters:",grid_search.best_params_)

y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Name: SURUTHIKA V")
print("Register Number: 212225040441")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
<img width="899" height="709" alt="image" src="https://github.com/user-attachments/assets/c8b5b9df-2e1a-4734-b04d-4466a69a0ca8" />
<img width="654" height="80" alt="image" src="https://github.com/user-attachments/assets/773a8027-7ed6-4dfb-a651-25cd3e333fba" />
<img width="638" height="286" alt="image" src="https://github.com/user-attachments/assets/e4f09f35-5362-4e83-8214-f60c952d834e" />
<img width="750" height="577" alt="image" src="https://github.com/user-attachments/assets/c398544b-442b-4b9f-a0bd-3fc8dd8a5bca" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
