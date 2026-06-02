# Ονοματεπώνυμο:    Mohammad Matin Marzie
# Αριθμός μητρώου:  inf2022001

# Εργασία 9

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score






# ----------Συνάρτηση για σχεδίαση περιοχών απόφασης----------
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    color=cmap(idx),
                    marker=markers[idx],
                    label=cl)
    
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    facecolors='none',
                    edgecolors='black',
                    s=55,
                    label='test set')
        



# --- Φόρτωση δεδομένων ---
X = np.loadtxt('./iris_data.txt', usecols=(2, 3))
Y = np.loadtxt('./iris_labels.txt')

# ----------Διαχωρισμός σε train/test----------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# ----------Ταξινομητής----------
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf.fit(X_train, Y_train)

# ----------Προβλέψεις και ακρίβεια----------
Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

print(f"Λανθασμένες προβλέψεις: {round((1 - accuracy) * 100, 2)}%")
print(f"Σωστές προβλέψεις: {round(accuracy * 100, 2)}%")

# ----------Συνδυασμένα δεδομένα για σχεδίαση----------
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((Y_train, Y_test))

# ----------Σχεδίαση περιοχών απόφασης----------
plot_decision_regions(X_combined, y_combined, classifier=clf, test_idx=range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Decision Tree (Entropy, max_depth=3)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# ----------5-fold cross-validation----------
scores = cross_val_score(clf, X, Y, cv=5)
print(f"Μέσο ποσοστό επιτυχίας με 5-fold CV: {round(np.mean(scores) * 100, 2)}%")
