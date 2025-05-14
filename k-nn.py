import pandas as pd

# Carga dataset
dataset = pd.read_csv('water_potability.csv')

dataset = dataset.dropna() # Elimina registros con valores nulos
X = dataset.iloc[:,:-1].values # variables independientes: todas las características
y = dataset.iloc[:,-1].values # objetivo: potability


# Divide dataset (train 75%, test 25%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrena modelo
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 21, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Clasifica el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evalúa modelo
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)
print("Precisión del modelo:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)