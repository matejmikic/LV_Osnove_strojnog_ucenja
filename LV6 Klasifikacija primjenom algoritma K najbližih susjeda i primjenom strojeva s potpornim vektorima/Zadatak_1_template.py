import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()



# Zadatak 6.5.1 - KNN s K=5
print("\n--- Zadatak 6.5.1 ---")
# KNN model s K=5
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_n, y_train)

# Evaluacija KNN modela
y_train_knn = knn_model.predict(X_train_n)
y_test_knn = knn_model.predict(X_test_n)

print("KNN (K=5): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn)))

# Granica odluke za KNN model s K=5
plot_decision_regions(X_train_n, y_train, classifier=knn_model)
plt.xlabel('Age (standardized)')
plt.ylabel('Estimated Salary (standardized)')
plt.legend(loc='upper left')
plt.title("KNN (K=5) - Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn)))
plt.tight_layout()
plt.show()

# KNN s K=1
knn_model_1 = KNeighborsClassifier(n_neighbors=1)
knn_model_1.fit(X_train_n, y_train)

# Evaluacija KNN modela s K=1
y_train_knn_1 = knn_model_1.predict(X_train_n)
y_test_knn_1 = knn_model_1.predict(X_test_n)

print("\nKNN (K=1): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn_1)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn_1)))

# Granica odluke za KNN model s K=1
plot_decision_regions(X_train_n, y_train, classifier=knn_model_1)
plt.xlabel('Age (standardized)')
plt.ylabel('Estimated Salary (standardized)')
plt.legend(loc='upper left')
plt.title("KNN (K=1) - Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn_1)))
plt.tight_layout()
plt.show()

# KNN s K=100
knn_model_100 = KNeighborsClassifier(n_neighbors=100)
knn_model_100.fit(X_train_n, y_train)

# Evaluacija KNN modela s K=100
y_train_knn_100 = knn_model_100.predict(X_train_n)
y_test_knn_100 = knn_model_100.predict(X_test_n)

print("\nKNN (K=100): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn_100)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn_100)))

# Granica odluke za KNN model s K=100
plot_decision_regions(X_train_n, y_train, classifier=knn_model_100)
plt.xlabel('Age (standardized)')
plt.ylabel('Estimated Salary (standardized)')
plt.legend(loc='upper left')
plt.title("KNN (K=100) - Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn_100)))
plt.tight_layout()
plt.show()







# Zadatak 6.5.2 - Cross-validation za optimalni K
print("\n--- Zadatak 6.5.2 ---")

from sklearn.model_selection import cross_val_score

# Definiranje raspona K vrijednosti
k_range = list(range(1, 31))
k_scores = []

# Izračun točnosti za različite K vrijednosti
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_n, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# Prikaz rezultata
plt.figure()
plt.plot(k_range, k_scores)
plt.xlabel('Vrijednost K')
plt.ylabel('Cross-validation točnost')
plt.title('Točnost KNN algoritma s različitim K vrijednostima')
plt.grid(True)
plt.show()

# Pronalazak optimalnog K
optimal_k = k_range[k_scores.index(max(k_scores))]
print(f"Optimalna vrijednost K: {optimal_k}")
print(f"Točnost za K={optimal_k}: {max(k_scores):.3f}")

# KNN model s optimalnim K
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train_n, y_train)

# Evaluacija KNN modela s optimalnim K
y_train_knn_opt = knn_optimal.predict(X_train_n)
y_test_knn_opt = knn_optimal.predict(X_test_n)

print(f"\nKNN (K={optimal_k}): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn_opt)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn_opt)))

# Granica odluke za KNN model s optimalnim K
plot_decision_regions(X_train_n, y_train, classifier=knn_optimal)
plt.xlabel('Age (standardized)')
plt.ylabel('Estimated Salary (standardized)')
plt.legend(loc='upper left')
plt.title(f"KNN (K={optimal_k}) - Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn_opt)))
plt.tight_layout()
plt.show()