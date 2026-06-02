# inf2022001
# Mohammad Matin Marzie
# Homeword 6

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def makePlotSVM(X_train,X_test,y_train,y_test,w,w0,left,right):
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
    y_minus=(-1-w0)/w[1]-(w[0]/w[1])*xl
    plt.plot(xl,y_minus)
    y_plus=(1-w0)/w[1]-(w[0]/w[1])*xl
    plt.plot(xl,y_plus)
#
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Perceptron Support Vector machine")
    plt.legend(loc="upper left")   
    plt.show()
##    



# Χρησιμοποιούμε τις δυο πρότες στύλες
X = np.loadtxt('./iris_data12.txt', usecols=(0, 1))
Y = np.loadtxt('./iris_labels12.txt')
Y[Y == 0] = -1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

clf = SVC(gamma='auto', kernel='linear', C=10.0)

clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

w0 = clf.intercept_[0]
w = clf.coef_[0]

# Δοκίμσα και για C==100.0 και C==10.0 έχω μόνο 2 βοηθητικά διανύσματα.

v1 = clf.support_vectors_[0]
v2 = clf.support_vectors_[1]

# Η απόσταση του βοηθητικού διανύσματος από την διαχωριστική ευθεία
d1 = abs(np.dot(w, v1) + w0) / np.linalg.norm(w)

d2 = abs(np.dot(w, v2) + w0) / np.linalg.norm(w)

print(f'd1: {d1: .4f}')
print(f'd2: {d2: .4f}')
print(f'd1+d2: {d1+d2: .4f}')
print(f'2/||w||: {2/np.linalg.norm(w): .4f}')


makePlotSVM(X_train,X_test,Y_train,Y_test,w,w0,left=4,right=8)