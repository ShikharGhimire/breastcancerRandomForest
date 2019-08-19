#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import KernelPCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from matplotlib import cm as cm
import seaborn as sns

#Importing the dataset
dataset = pd.read_csv('data.csv')
dataset.head()
dataset.shape     #Checking the dimensionality of the data
print("The dataset dimensionality is {}".format(dataset.shape))
dataset.drop('id',axis = 1,inplace = True) #Drop the ID independent variable
dataset.drop('Unnamed: 32',axis = 1,inplace = True) #Drop the "Unnamed 32" independent variable
dataset.head()
dataset.plot(kind = 'density',subplots = True, layout = (5,7), sharex = False, legend = False, fontsize = 1) #To plot the density of the graph
plt.show()
#Checking the correlation
f,ax = plt.subplots(figsize = (32,32))
sns.heatmap(dataset.corr(),annot = True, linewidths = 1.5, fmt = '.2f',ax=ax)
plt.show()
#Seperating the dataset
X = dataset. iloc[:,1:30].values
Y = dataset.iloc[:,0].values
#Visualising how many data are melignant and how many are benign
sns.countplot(Y,label = 'Number of tumors(Blue = Benign tumor and Orange= Malignant tumor)')
#Categorically changing the data for Y ( 0 is malignant and 1 is beneign)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)
#Splitting the dataset into the training set and testing data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)
#Feature scaling
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 9,criterion = 'entropy',max_depth = 4, random_state = 0)

reduction = KernelPCA(n_components=2, kernel = 'linear')
x_train_reduced = reduction.fit_transform(X_train) #Reducing the data into 2 components
x_test_reduced = reduction.transform(X_test) #Reducing it to 3D into components
classifier.fit(x_train_reduced, y_train) #Fitting the x_train and the y_train

#Predicting the X test
y_pred = classifier.predict(x_test_reduced)
# Using the confusion matrix 
cm = confusion_matrix(y_test, y_pred)

#Applying K-fold cross validation to evaluate the model performance
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train_reduced, y = y_train,cv = 10, n_jobs = -1) #10 fold cross validation
accuracies.mean()
accuracies.std()

#Checking whether the data can be classified using random forest linear model or non linear model
#Finding the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [1,2,3,4,5,6,7,8,9,5,10,20,30,40,50],
               'max_features': ['auto', 'sqrt', 'log2'],
               'max_depth' : [1,2,3,4,5,6,7,8],
               'criterion' :['gini', 'entropy']}]
gridsearch = GridSearchCV(estimator = classifier,param_grid = parameters, 
                          scoring = 'accuracy',cv = 10,n_jobs = -1)

#Fitting gridsearch in training set
grid_search = gridsearch.fit(x_train_reduced,y_train)
#Displaying the optimnal parameters using grid_search
best_accuracy = gridsearch.best_score_ #Accuracy percentage
best_parameters = gridsearch.best_params_


X_set, y_set = x_train_reduced, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Detection of Cancer in Training set')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()

#Test set boundary
X_set, y_set = x_test_reduced, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Detection of Cancer in Testing set')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()

