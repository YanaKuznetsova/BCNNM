import csv
import functools
import math
import scipy

import numpy as np
import pandas as pd
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

from time import time

timeM = {}

def startTime(i):
    global timeM
    timeM[i] = time()

def printTime(i, text):
    global timeM
    print("\t"*i + text + " " +str(time()-timeM[i]))

###################################################################################
# подготовка данных
###################################################################################
def getData (filename):

    init = pd.read_csv(filename)
    init = np.array(init)
    print('--> Data have been read <--' )
    X = modifyData(init, filename)
    print('--> Data have been modified <--')
    X = addFeatures(X, filename)
    print('--> Features have been added <--')
    X, Y = selectXY(X)
    Y = np.array([Y[i,0] for i in range(Y.shape[0])])
    return X, Y

def modifyData(matrix, filename):
    # удаление неинформативных строк и строк с нулями
    matrix = matrix[:, 0:4]
    matrix = matrix[matrix[:, 1:].all(1)]

    # проекция на плоскость
    def plane(x, y, params):
        a = params[0]
        b = params[1]
        c = params[2]
        z = a * x + b * y + c
        return z

    def error(params, points):
        result = 0
        for (x, y, z) in points:
            plane_z = plane(x, y, params)
            diff = abs(plane_z - z)
            result += diff ** 2
        return result

    # находим плоскость, в которой лежат точки
    fun = functools.partial(error, points=matrix[:,1:])
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)

    # уравнение плоскости  ax + by + c = z
    a = res.x[0]
    b = res.x[1]
    c = res.x[2]
    # нормаль: [a, b, -1]
    norm = np.array([a, b, -1])
    normLen = math.sqrt((norm ** 2).sum())
    norm /= normLen
    xbasis = np.array([0, 0, c])
    xbasis /= math.sqrt((xbasis ** 2).sum())
    ybasis = np.cross(xbasis, norm)

    newMatrix = np.array([[point[0], np.dot(point[1:],xbasis), np.dot(point[1:],ybasis), np.dot(point[1:],norm)-c/normLen] for point in matrix])

    # находим центр масс для перехода в полярные координаты
    # первая интерация: все массы одинаковы
    # вторая итерация: масса обратно пропорциональа растоянию до ц.м.№1
    # последняя итерация: находим точку, ближайшую к ц.м.№1 и ц.м.№2
    r,c = newMatrix.shape
    massX = np.array(np.ones(r))
    massY = np.array(np.ones(r))
    cgx1 = np.sum(newMatrix[:, 1] * massX) / np.sum(massY)
    cgy1 = np.sum(newMatrix[:, 2] * massX) / np.sum(massY)
    for i in range (r):
        massX[i] = 1/abs(newMatrix[i,1] - cgx1)
        massY[i] = 1/abs(newMatrix[i,2] - cgy1)
    cgx2 = np.sum(newMatrix[:, 1] * massX) / np.sum(massY)
    cgy2 = np.sum(newMatrix[:, 2] * massX) / np.sum(massY)

    minDist = 1000
    for i in range(r):
        dist = (newMatrix[i,1] - cgx1)**2 + (newMatrix[i,2] - cgy1)**2
        dist += (newMatrix[i,1] - cgx2)**2 + (newMatrix[i,2] - cgy2)**2
        if (dist < minDist):
            cgx = newMatrix[i,1]
            cgy = newMatrix[i,2]
            minDist = dist
    #print('The center of mass in x is %f' % cgx)
    #print('The center of mass in y is %f' % cgy)

    # переход в полярные координаты
    newMatrix = np.array([ [point[0], point[1] - cgx, point[2] - cgy, point[3]] for point in newMatrix])
    newMatrix = np.array([[point[0], math.sqrt(point[1]**2 + point[2]**2), math.atan(point[2]/point[1]), point[3]] for point in newMatrix if point[1] != 0 and point[2] != 0])

    # сохранение файла с данными
    #filename = filename[:-4] + '_mod3.dat'
    #getMatrix(newMatrix, filename)
    return(newMatrix)

def addFeatures(matrix,filename):

    # добавление новых фичей
    A2 = np.matrix(matrix[:, 1] * matrix[:, 1]).transpose()
    B2 = np.matrix(matrix[:, 2] * matrix[:, 2]).transpose()
    C2 = np.matrix(matrix[:, 3] * matrix[:, 3]).transpose()
    A3 = np.matrix(matrix[:, 1] * matrix[:, 1]* matrix[:, 1]).transpose()
    B3 = np.matrix(matrix[:, 2] * matrix[:, 2]* matrix[:, 2]).transpose()
    C3 = np.matrix(matrix[:, 3] * matrix[:, 3] * matrix[:, 3]).transpose()
    AB = np.matrix(matrix[:, 1] * matrix[:, 2]).transpose()
    AC = np.matrix(matrix[:, 1] * matrix[:, 3]).transpose()
    BC = np.matrix(matrix[:, 2] * matrix[:, 3]).transpose()
    eA = np.matrix(np.exp(matrix[:, 1])).transpose()
    eB = np.matrix(np.exp(matrix[:, 2])).transpose()
    eC = np.matrix(np.exp(matrix[:, 3])).transpose()
    thA = np.matrix(np.tanh(matrix[:, 1])).transpose()
    thB = np.matrix(np.tanh(matrix[:, 2])).transpose()
    thC = np.matrix(np.tanh(matrix[:, 3])).transpose()
    tgB = np.matrix(np.tan(matrix[:, 2])).transpose()

    matrix = np.hstack((matrix, A2, B2, C2, A3, B3, C3, AB, BC, AC, eA, eB, eC, thA, thB, thC, tgB))

    # сохранение файла с данными
    #filename = filename[:-3] + 'dat'
    #getMatrix(matrix, filename)

    return matrix

def selectXY(matrix):
    X = matrix[:, 1:]
    Y = matrix[:, 0]
    return X, Y

###################################################################################
# анализ данных
###################################################################################

def k_means_clustering(X, Y):
    from sklearn.neighbors import KNeighborsClassifier as KNN

    tmp = X.astype(float)
    X_scaled = scale(X)

    print("K-means clustering" )
    param_grid = [{'n_neighbors': [3, 5, 7, 9, 11, 20, 30, 50]}]
    searcher = GridSearchCV(KNN(), param_grid, cv=2)
    searcher.fit(X_scaled, Y)
    print(searcher.best_estimator_)
    scores = cross_val_score(searcher.best_estimator_, X_scaled, Y, cv = 10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

def hierarchical_clustering(X, Y):
    from sklearn.cluster import AgglomerativeClustering

    print("Hierarchical clustering" )
    param_grid = [{'linkage': ['average', 'complete',  'ward'],
                   'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'] }]
    searcher = GridSearchCV(AgglomerativeClustering(n_clusters=2, memory='mycachedir', compute_full_tree=True), param_grid, n_jobs=-1, verbose=2, scoring='adjusted_rand_score')
    searcher.fit(X)
    print(searcher.best_estimator_)
    scores = cross_val_score(searcher.best_estimator_, X, Y, cv=10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

def birch_cluster(X, Y):
    from sklearn.cluster import Birch

    print("Birch clustering" )
    param_grid = [{'threshold': [0.05, 0.1,0.2,0.3,0.5],
                   'n_clusters': [2, 6 ,8, 10, 15, 20]}]
    searcher = GridSearchCV(Birch(n_clusters=2), param_grid, scoring='adjusted_rand_score')
    searcher.fit(X,Y)
    print(searcher.best_estimator_)
    scores = cross_val_score(searcher.best_estimator_, X, Y, cv=10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

def bayes_classifier(X, Y):
    from sklearn.naive_bayes import GaussianNB

    X_scaled = scale(X)

    print("Bayes classifier:")
    model = GaussianNB()
    scores = cross_val_score(model, X_scaled, Y, cv=10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

def decision_tree(X, Y):
    from sklearn.tree import DecisionTreeClassifier

    print("Decision tree:")

    param_grid = [{'max_depth': [2, 3, 5, 10], 'min_samples_leaf': [1, 3, 7, 10, 15]}]
    searcher = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    searcher.fit(X, Y)
    print(searcher.best_estimator_)
    scores = cross_val_score(searcher.best_estimator_, X, Y, cv=10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

###################################################################################
def getResult(result, n, filename):
    with open(filename, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow(['k', 'score'])
        for i in range(n):
            wr.writerow([i+3, result[i]])
    return 0

def getMatrix(matrix, filename):
    with open(filename, 'w', newline='') as csvfile:
        r,c = matrix.shape
        wr = csv.writer(csvfile, delimiter=',')
        for i in range(r):
            wr.writerow(matrix[i, :])
    return 0

############################################################
filenames =  ('1486416673294.csv', '1486430023690.csv', '1486435241183.csv', '1486444008135.csv', '1486448918687.csv')
for f in filenames:
     X, Y  = getData(f)
     print(f)
     bayes_classifier(X, Y)
     decision_tree(X, Y)
     print ("\n------------------------------------\n")


# X, Y = getData('1486416673294.csv')
#
# #hierarchical_clustering(X, Y)
# #birch_cluster(X, Y)
# bayes_classifier(X, Y)
# decision_tree(X, Y)
# k_means_clustering(X,Y)
print("--->>>Done!<<<---")

