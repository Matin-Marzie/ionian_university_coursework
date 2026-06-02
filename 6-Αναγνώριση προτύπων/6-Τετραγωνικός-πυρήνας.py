# Ονοματεπώνυμο: Mohammad Matin Marzie
# Αριθμός μητρώου: inf2022001

import numpy as np
import matplotlib.pyplot as plt


def perceptron_training_square_kernel(X, samples_labels):
    lamda_j = np.zeros(len(X))

    # Δείκτης για να καθορίσουμε πότε βρέθηκε η ευθεία
    found_line = False
    while not found_line:
        found_line = True
        for index, (x_i, y_i) in enumerate(zip(X, samples_labels)):

            kernel = np.sum(lamda_j * (( 1 + np.dot(X, x_i)) **2))

            if y_i * kernel <= 0:
                lamda_j[index] += y_i
                found_line = False
    return lamda_j





def makePlot_xordata_polyn_kernel(X_xor, y_xor, X_train, a):
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
    plt.ylim(-3.0)

    xl, yl = [], []
    x = np.linspace(-2, 2, 401)
    y = np.linspace(-3, 2, 501)
    
    for i in range(len(x)):
        for j in range(len(y)):
            ll = np.array([x[i], y[j]])
            s = sum(a[j_sample] * (1 + np.dot(ll, X_train[j_sample, :]))**2 for j_sample in range(len(X_train)))
            if np.abs(s) < 0.1:
                xl.append(x[i])
                yl.append(y[j])

    sizes = np.full(len(xl), 5)
    plt.scatter(xl, yl, sizes)
    plt.legend()
    plt.title("Perceptron - τετραγωνικός πυρήνας")
    plt.show()



# ||---------- MAIN PROGRAM ----------||
# Δημιουργία XOR δεδομένων
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# Ανακάτεμα και κατάτμηση δεδομένων (75% εκπαίδευση, 25% έλεγχος)
samples = np.column_stack((X_xor, y_xor))
samples = samples[np.random.permutation(len(samples))]
training_sample, testing_sample = np.split(samples, [int(0.75 * len(samples))])

# Διαχωρισμός X_train, X_test, y_train, y_test
X_train = training_sample[:, :-1]
Y_train = training_sample[:, -1]
X_test = testing_sample[:, :-1]
Y_test = testing_sample[:, -1]

# Εκπαίδευση του Perceptron
lamda_j = perceptron_training_square_kernel(X_train, Y_train)
# lamda_j = np.ones(len(X_train))

# Κλήση της συνάρτησης σχεδίασης
makePlot_xordata_polyn_kernel(X_xor, y_xor, X_train, lamda_j)

# Βρίσκουμε το πλήθος των λανθασμένων προβλέψεων
wrong_predicted_count = 0
for x_test, y_test in zip(X_test, Y_test):
    k = (1 + np.dot(X_train, x_test))**2
    y_pred = np.sign(np.sum(lamda_j * k))
    if y_pred != y_test:
        wrong_predicted_count += 1

# Υπολογισμός των σωστών προβλέψεων
correct_predicted_count = len(X_test) - wrong_predicted_count

# Υπολογισμός των ποσοστών
wrong_predicted_percentage = int((wrong_predicted_count / len(X_test)) * 100)
correct_predicted_percentage = int((correct_predicted_count / len(X_test)) * 100)

print(f"{'Λανθασμένες προβλέψεις:':<25} {wrong_predicted_percentage}%")
print(f"{'Σωστές προβλέψεις:':<25} {correct_predicted_percentage}%")
