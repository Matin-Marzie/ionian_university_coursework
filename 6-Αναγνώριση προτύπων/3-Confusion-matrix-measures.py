import numpy as np
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

breast_cancer_samples = datasets.load_breast_cancer()

# Κρατάμε μόνο τα δυο πρώτα χαρακτηριστικά
X = breast_cancer_samples.data[:, :2]
Y = breast_cancer_samples.target

#  Χωρίζουμε σε 75% 25%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# τιμές των μέτρων Ορθότητας
accuracies = []

# Εκτελώ τον αλγόριθμο κοντινότερων γειτόνων για διάφορες τιμες του k
for k in range(1, 16):

    neigh = neighbors.KNeighborsClassifier(n_neighbors=k)

    neigh.fit(X_train, Y_train)

    Y_pred = neigh.predict(X_test)

    cm = confusion_matrix(Y_test, Y_pred)
    
    accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
    accuracies.append(accuracy)


# ο δείκτης του πίνακα που ενέχει την μεγαλύτερη ορθότητα
max_arg_index = np.argmax(accuracies)
optimal_k = max_arg_index + 1

neigh = neighbors.KNeighborsClassifier(n_neighbors=optimal_k)
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred) # Documentation: Thus in binary classification, the count of true negatives is {0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}

# Για την τιμή που επιτυγχάνει μεγαλύτερη ορθότητα τυπώνουμε τις τιμές:
print(f"----------Για κ = {optimal_k}:----------")
# της ορθότητας
print(f"{'accuracy:': <25} {round(accuracies[max_arg_index], 3)}")
#  της ακρίβειας
precision = round((cm[1][1]) / (cm[1][1] + cm[0][1]), 3)
print(f"{'precision:': <25} {precision}")
#  της ανάκλησης
recall = round((cm[1][1]) / (cm[1][1] + cm[1][0]), 3)
print(f"{'recall:': <25} {recall}")


# Σχεδίαση με plot τον πίνακα σύγχυσης
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# accuracy = TP + TN / TP + FP + TN + FN
# accuracy = 85 + 44 / 85 + 10 + 44 + 04 = 129 / 143 = 0.902

# precision = TP / TP + FP
# precision = 85 / 85 + 10 = 85 / 95 = 0.894

# recall = TP / TP + FN
# recall = 85 / 85 + 4 = 85 / 89 = 0.955