import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel() # popravljeno: .ravel() da y bude vektor

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

bars_train = np.unique(y_train, return_counts=True)
bars_test = np.unique(y_test, return_counts=True)

plt.figure()  
plt.bar(bars_train[0] - 0.2, bars_train[1], width=0.4, label='Training')
plt.bar(bars_test[0] + 0.2, bars_test[1], width=0.4, label='Testing')
plt.xticks(np.arange(3), ["Adelie", "Chinstrap", "Gentoo"])
plt.ylabel('Broj primjera')
plt.legend()
plt.title('Broj primjera po vrsti pingvina')
plt.show()


model = LogisticRegression(multi_class="multinomial", solver='lbfgs', max_iter=1000) 
model.fit(X_train, y_train)


print("Intercept (theta_0):", model.intercept_)
print("Coefficients (theta_1, theta_2,...):", model.coef_)


plot_decision_regions(X_train, y_train, model)
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.title('Granice odluke logističke regresije')
plt.show()


y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Adelie", "Chinstrap", "Gentoo"]))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Adelie", "Chinstrap", "Gentoo"]).plot()
plt.title('Matrica zabune')
plt.show()


input_variables_full = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X_full = df[input_variables_full].to_numpy()
y_full = df[output_variable].to_numpy().ravel()

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size = 0.2, random_state = 123)

logi_reg_full = LogisticRegression(multi_class="multinomial", solver='lbfgs', max_iter=1000)
logi_reg_full.fit(X_train_full, y_train_full)

y_pred_full = logi_reg_full.predict(X_test_full)

print("Classification Report with additional features:\n", classification_report(y_test_full, y_pred_full, target_names=["Adelie", "Chinstrap", "Gentoo"]))

ConfusionMatrixDisplay(confusion_matrix(y_test_full, y_pred_full), display_labels=["Adelie", "Chinstrap", "Gentoo"]).plot()
plt.title('Matrica zabune (s dodatnim značajkama)')
plt.show()
