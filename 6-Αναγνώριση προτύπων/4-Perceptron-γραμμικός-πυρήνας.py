# Ονοματεπώνυμο:    Mohammad Matin Marzie
# Αριθμός μητρώου:  inf2022001

# Εργασία 2

import numpy as np
from sklearn import datasets

# Δείγματα
X = np.loadtxt('./iris_data12.txt', usecols=(0, 2))
# Ετικέτες
samples_labels = np.loadtxt('./iris_labels12.txt')

iris = datasets.load_iris()


# ----------Convernt 0 to -1 in labels----------
# Using list comprehension
# samples_labels = [-1 if label == 0 else 1 for label in samples_labels]

# Using NumPy Vectorized
# Δεν δημιουργεί επιπλέιον λίστα και ως τέτοια, καταναλώνεται λιγότερη μνήμη
samples_labels[samples_labels == 0] = -1

def perceptron_training(X, samples_labels):
    w0 = 10
    lamda_j = np.zeros(len(X))

    # Δείκτης για να καθορίσουμε πότε βρέθηκε η ευθεία
    found_line = False
    while not found_line:
        found_line = True
        for index, (x_i, y_i) in enumerate(zip(X, samples_labels)):
            # y(i) ( Σ lamda_j * x(j) * x(i) + W0) <= 0
            if y_i * (np.dot( np.sum(lamda_j[:, None] * X, axis=0)  , x_i) + w0)<= 0:
                # w = w + y(i) * x(i)
                lamda_j[index] += y_i
                # W0 = W0 + x(i)
                w0 += y_i
                found_line = False
    return lamda_j, w0



def makePlotPerc_final(X_train,X_test,y_train,y_test,w,w0,left,right):
    import numpy as np
    import matplotlib.pyplot as plt
    x_pos=[]
    y_pos=[]
    x_neg=[]
    y_neg=[]
    for i in range(len(X_train)):    
        if y_train[i] == 1:
           x_pos.append(X_train[i,0])
           y_pos.append(X_train[i,1])
        else:
           x_neg.append(X_train[i,0])
           y_neg.append(X_train[i,1])
    plt.scatter(x_pos, y_pos, c="g",label="versicolor")
    plt.scatter(x_neg, y_neg, c="r",label="setosa")       
#
    x1_pos=[]
    y1_pos=[]
    x1_neg=[]
    y1_neg=[]
    for i in range(len(X_test)):    
        if y_test[i] == 1:
           x1_pos.append(X_test[i,0])
           y1_pos.append(X_test[i,1])
        else:

           x1_neg.append(X_test[i,0])
           y1_neg.append(X_test[i,1]) 
    plt.scatter(x1_pos, y1_pos, c="g",marker="+")
    plt.scatter(x1_neg, y1_neg, c="r",marker="+")           
    xl=np.linspace(left,right,10)
    yl=(-w0/w[1])-(w[0]/w[1])*xl
    plt.plot(xl,yl)
#
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("perceptron")
    plt.legend(loc="upper left")
    plt.show()
#
#

# ||----------MAIN-PROGRAM----------||

# Ένωση των δειγμάτων με τις ετικέτες τους έτσι ώστε όταν ανακατεύω(shuffle) τα δείγματα πριν την κατάτμηση(75%,25%) να μην χαθεί η αντιστοιχεία
# samples: [x1, x2, y]
samples = np.column_stack((X, samples_labels))


# Πρώτα ανακατεύουμε τα δείγματά μας και έπειτα χωρίζουμε στα 75% και 25%
samples = samples[np.random.permutation(len(samples))]
training_sample, testing_sample = np.split(samples, [int(0.75 * len(samples))])

# Μετατροπή σε μορφή: X_train, X_test, y_train, y_test
X_train = training_sample[:, :-1] # όλες οι γραμμές, όλες οι στύλες εκτός την τελευταία
Y_train = training_sample[:, -1]  # όλες οι γραμμές, και η τελευταία στύλη
X_test = testing_sample[:, :-1]
Y_test = testing_sample[:, -1]


lamda_j, w0 = perceptron_training(X_train, Y_train)

# Μετατροπή του λ σε w
w = np.sum(lamda_j[:, None] * X_train, axis=0)

makePlotPerc_final(X_train, X_test, Y_train, Y_test, w, w0, left=4, right=8)

# Βρίσκουμε το πλήθος των λανθασμένων προβλέψεων
wrong_predicted_count = 0
for x_test, y_test in zip(X_test, Y_test):
    y_pred = np.sign(np.sum(lamda_j * np.dot(X_train, x_test)) + w0)
    if y_pred != y_test:
        wrong_predicted_count +=1

# Πλήθος των σωστών προβλέψεων
correct_predicted_count = len(X_test) - wrong_predicted_count

# Υπολογισμός των ποσοστών
wrong_predicted_percentage = int((wrong_predicted_count/len(X_test)) * 100)
correct_predicted_percentage = int((correct_predicted_count/len(X_test)) * 100)

print(f"{'Λανθασμένες προβλέψεις:':<25} {wrong_predicted_percentage}%")
print(f"{'Σωστές προβλέψεις:':<25} {correct_predicted_percentage}%")

