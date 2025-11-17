from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from utils import load_data, load_weights,one_hot_encoding, accuracy
from public_test import checkNNGradients,MLP_test_step
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier



"""
Test 1 to be executed in Main
"""
def gradientTest():
    checkNNGradients(costNN,target_gradient,0)
    checkNNGradients(costNN,target_gradient,1)


"""
Test 2 to be executed in Main
"""
def MLP_test(X_train,y_train, X_test, y_test):
    print("We assume that: random_state of train_test_split  = 0 alpha=1, num_iterations = 2000, test_size=0.33, seed=0 and epislom = 0.12 ")
    print("Test 1 Calculando para lambda = 0")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,0,2000,0.92606,2000/10)
    print("Test 2 Calculando para lambda = 0.5")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,0.5,2000,0.92545,2000/10)
    print("Test 3 Calculando para lambda = 1")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,1,2000,0.92667,2000/10)

def MLP_test_SKLearn(X_train,y_train, X_test, y_test):
    print("Test de SKLearn")
    print("Test 1 Calculando para lambda = 0")
    clf = MLPClassifier(activation='logistic', verbose=False, alpha= 0,random_state=0, max_iter=2000,epsilon=0.12,learning_rate_init=0.01).fit(X_train, y_train)
    score = clf.score(X_test,y_test)
    print("Accuracy de sklearn classifier con lambda = 0:", score)

    print("Test 2 Calculando para lambda = 0.5")
    clf = MLPClassifier(activation='logistic', verbose=False, alpha= 0.5,random_state=0, max_iter=2000,epsilon=0.12,learning_rate_init=0.01).fit(X_train, y_train)
    score = clf.score(X_test,y_test)
    print("Accuracy de sklearn classifier con lambda = 0.5:", score)

 
    print("Test 3 Calculando para lambda = 1")
    clf = MLPClassifier(activation='logistic', verbose=False , alpha= 1,random_state=0, max_iter=2000,epsilon=0.12,learning_rate_init=0.01).fit(X_train, y_train)
    score = clf.score(X_test,y_test)
    print("Accuracy de sklearn classifier con lambda = 1:", score)



def main():
#TO-DO: calculate a testing a prediction and cost.
    print("Main program")
    
    #Test 1
    gradientTest()

    ## TO-DO: descoment both test and create the needed code to execute them.
    x,y = load_data('data/ex3data1.mat')
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

    #Pasamos los datos de salida de entrenamiento a one hot encoding
    y_train_OneHot = one_hot_encoding(y_train)

    #Test 2
    MLP_test(x_train,y_train_OneHot, x_test, y_test)

    #Test con MLPClassifier (hay que pasar y_train sin formato One_Hot_Encoding)
    MLP_test_SKLearn(x_train,y_train, x_test, y_test)
    
main()