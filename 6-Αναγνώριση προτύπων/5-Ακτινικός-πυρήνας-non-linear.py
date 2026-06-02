# Ονοματεπώνυμο: Mohammad Matin Marzie
# Αριθμός μητρώου: inf2022001


# ----------------PLOT-------------------
def makePlotPerc_kernel(X_train,X_test,y_train,y_test,a,left=0,right=0):
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
    plt.scatter(x_pos, y_pos, c="g",label="pos")
    plt.scatter(x_neg, y_neg, c="r",label="neg")       
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
    plt.scatter(x1_pos, y1_pos, c="g",marker="*")
    plt.scatter(x1_neg, y1_neg, c="r",marker="*")
#################    
    xl=[]
    yl=[]
    x = np.linspace(-3, 3, 801)
    y = np.linspace(5, 10, 251)
    for i in range(len(x)):
        for j in range(len(y)):
            ll=[0,0]
            s=0
            ll[0]=x[i]
            ll[1]=y[j]
            ll=np.array(ll)
            for j_sample in range(len(X_train)):
#               s += a[j_sample]*(1+ np.dot(ll,X_train[j_sample,:]))**2
                s += a[j_sample]*np.exp(-np.dot((ll-X_train[j_sample,:]),(ll-X_train[j_sample,:])))
            if(np.abs(s) < 0.001):
              xl.append(x[i])
              yl.append(y[j])
#################               
#    xl=np.linspace(left,right,10)
#    yl=(-w[0]/w[2])-(w[1]/w[2])*xl
#   print(len(xl))
    plt.scatter(xl,yl)
#
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("ακτινικός πυρήνας - non linearly separable data")
    plt.legend(loc="upper left")
    plt.show()
#





import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split






# ||------------------TRAIN-PERCEPTRON----------------------||

# ----------Με ακτινικό πυρήνα----------

def perceptron_training(X, samples_labels):
    lamda_j = np.zeros(len(X))

    # Δείκτης για να καθορίσουμε πότε βρέθηκε η ευθεία
    found_line = False
    while not found_line:
        found_line = True
        for index, (x_i, y_i) in enumerate(zip(X, samples_labels)):

            kernel = np.sum(lamda_j * np.exp(-(np.linalg.norm(X - x_i, axis=1)**2)))

            if y_i * kernel <= 0:
                lamda_j[index] += y_i
                found_line = False
    return lamda_j




# ||---------- MAIN PROGRAM ----------||


X = np.loadtxt('perceptron_non_separ.txt', usecols=(0, 1))
Y = np.loadtxt('perceptron_non_separ.txt', usecols=(2))
print('perceptron_non_separ.txt loaded...')

#  Χωρίζουμε σε 75% 25%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Εκπαίδευση του Perceptron
lamda_j = perceptron_training(X_train, Y_train)
print('training completed...')


# Κλήση της συνάρτησης σχεδίασης
print('Creating plot...')
makePlotPerc_kernel(X_train,X_test,Y_train,Y_test,a=lamda_j)

# Βρίσκουμε το πλήθος των λανθασμένων προβλέψεων
wrong_predicted_count = 0
for x_test, y_test in zip(X_test, Y_test):
    kernel = np.sum(lamda_j * np.exp(-(np.linalg.norm(X_train - x_test, axis=1)**2)))
    y_pred = np.sign(kernel)
    if y_pred != y_test:
        wrong_predicted_count += 1

# Υπολογισμός των σωστών προβλέψεων
correct_predicted_count = len(X_test) - wrong_predicted_count

# Υπολογισμός των ποσοστών
wrong_predicted_percentage = int((wrong_predicted_count / len(X_test)) * 100)
correct_predicted_percentage = int((correct_predicted_count / len(X_test)) * 100)

print(f"{'Λανθασμένες προβλέψεις:':<25} {wrong_predicted_percentage}%")
print(f"{'Σωστές προβλέψεις:':<25} {correct_predicted_percentage}%")
