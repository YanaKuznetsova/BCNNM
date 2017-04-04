import os
import tarfile
import csv
import functools
import math
import scipy
from scipy import optimize
import numpy as np
import pandas as pd
import statistics as stat
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

def getData (file, filename):
    init = pd.read_csv(file)
    init = np.array(init)
    #print('--> Data have been read <--' )
    modifyData(init, filename)

def modifyData(matrix, filename):
    # 0 - Label
    # 1,2,3 - DecisionNeurons
    # 4 - AnswerLength
    # 5 - Tick
    # 6 - velocity
    # 7 - start of new data

    (r,c) = matrix.shape
    velocity = np.zeros(r)
    times = np.zeros(r)
    for i in range (r):
        if (i == 0):
            velocity[i] = matrix[0, 5] - matrix[0, 4]
        else:
            velocity[i] = matrix[i, 5] - matrix[i-1, 5] - matrix[i, 4]
            times[i] = matrix[i-1,5]
    new_matrix = np.column_stack((matrix, velocity, times))

    # удаляем столбцы с одними нулями
    #new_matrix = new_matrix[new_matrix[:, 1:4].all(1)]

    # проекция в плоскость, переход к полярной системе координат
    #new_matrix = move_to_plane(new_matrix)

    # время ответа, скорость ответа, кол-во ошибок, расстояние между центрами масс, среднее удаление от ц.м.
    #calc_quality(new_matrix, filename)

    # время ответа, скорость ответа, кол-во ошибок, расстояние между центрами масс, среднее удаление от ц.м.
    # рассматриваем только с конца: последние 10, 15, 20, 25, 50% датасета
    calc_quality_end(new_matrix, filename)

    # время ответа, скорость ответа, средние значения нейронов
    #calc_statistic(new_matrix, filename)

    # разбиваем данные по классам
    #ones = new_matrix[new_matrix[:,0] == 1]
    #zeros = new_matrix[new_matrix[:,0] == 0]
    # время ответа, скорость ответа, средние значения нейронов
    #calc_statistic(zeros, filename + '_zeros')
    #calc_statistic(ones, filename + '_ones')
    return (new_matrix)

def calc_quality(matrix, filename):
    def calc_mistakes(matrix): #  расчет кол-ва одинаковых ответов в матрице при условии разных данных на вход
        new_matrix = sorted(matrix, key=lambda data: data[0])
        new_matrix = sorted(new_matrix, key=lambda data: data[1])
        new_matrix = sorted(new_matrix, key=lambda data: data[2])
        new_matrix = sorted(new_matrix, key=lambda data: data[3])
        new_matrix = np.array(new_matrix)
        r,c = new_matrix.shape
        cnt = 0
        label = new_matrix[0, 0]
        tmp1 = new_matrix[0, 1]
        tmp2 = new_matrix[0, 2]
        tmp3 = new_matrix[0, 3]
        for i in range (1,r):
            if ((label != new_matrix[i,0]) &
                (new_matrix[i, 1] == tmp1) & (new_matrix[i,2] == tmp2) & (new_matrix[i,3] == tmp3)):
                cnt += 1
            else:
                label = new_matrix[i, 0]
                tmp1 = new_matrix[i, 1]
                tmp2 = new_matrix[i, 2]
                tmp3 = new_matrix[i, 3]
        #print("cnt " + str(cnt))
        return cnt

    def calc_distance(matrix):
        ones = matrix[matrix[:, 0] == 1]
        zeros = matrix[matrix[:, 0] == 0]
        cgx_o, cgy_o, cgz_o = calc_gravity_center(ones)
        cgx_z, cgy_z, cgz_z = calc_gravity_center(zeros)
        #print("distance" , math.sqrt((cgx_o-cgx_z)**2 + (cgy_o-cgy_z)**2 + (cgz_o - cgz_z)**2 ))
        dist = math.sqrt((cgx_o - cgx_z) ** 2 + (cgy_o - cgy_z) ** 2 + (cgz_o - cgz_z) ** 2)
        radiusO = calc_radius (cgx_o, cgy_o, cgz_o, ones)
        radiusZ = calc_radius(cgx_z, cgy_z, cgz_z, zeros)
        return dist, radiusO, radiusZ

    def calc_radius(x,y,z,matrix):
        r, c = matrix.shape
        radius = 0
        for i in range(r):
            radius += math.sqrt((matrix[i,1] - x)**2 + (matrix[i,2] - y)**2 +(matrix[i,3] - z)**2 )
        radius /= r
        return radius


    def calc_gravity_center(matrix):
        r, c = matrix.shape
        mass = np.array(np.ones(r))
        cgx = np.sum(matrix[:, 1] * mass) / np.sum(mass)
        cgy = np.sum(matrix[:, 2] * mass) / np.sum(mass)
        cgz = np.sum(matrix[:, 3] * mass) / np.sum(mass)
        #print("cgx, cgy, cgz")
        #print(cgx, cgy, cgz)
        return (cgx, cgy, cgz)

    # 0 - Label
    # 1,2,3 - DecisionNeurons
    # 4 - AnswerLength
    # 5 - Tick
    # 6 - velocity
    # 7 - start of new data

    r,c = matrix.shape

    set_size = math.floor(r/10)-1
    step = math.floor(r/20)-1
    result = np.zeros(shape=(19, 13))

    for i in range(0,19):
        start = i*step
        finish = i*step + set_size
        result[i, 0] = matrix[start, 7] # time
        result[i, 1] = stat.median(matrix[start:finish, 4]) # average_answer_length
        result[i, 2] = stat.stdev(matrix[start:finish, 4])
        result[i, 3] = stat.median(matrix[start:finish, 6])  # average_velocity
        result[i, 4] = stat.stdev(matrix[start:finish, 6])
        result[i, 5] = calc_mistakes(matrix[start:finish, 0:4]) # number of mistakes
        dist, radiusO, radiusZ = calc_distance(matrix[start:finish, 0:4])
        result[i, 7] = dist # average distance between center of mass
        result[i, 9] = radiusO # average radius of ONES from center of mass
        result[i, 11] = radiusZ  # average radius of ZEROS from center of mass

    for i in range(0,19):
        result[i, 6] = result[i,5] / max(result[:,5]) # NORMED number of mistakes
        result[i, 8] = result[i,7] / max(result[:,7]) # NORMED average distance between center of mass
        result[i, 10] = result[i, 9] / max(result[:, 9])  # NORMED average radius of ONES from center of mass
        result[i, 12] = result[i, 11] / max(result[:, 11])  # NORMED average radius of ZEROS from center of mass

    write_data_quality(result, 19, filename+'_qual3')

    a0 = stat.median(result[:, 0])
    a1 = stat.median(result[:, 1])
    a2 = stat.median(result[:, 2])
    a3 = stat.median(result[:, 3])
    a4 = stat.median(result[:, 4])
    a5 = stat.median(result[:, 5])
    a6 = stat.median(result[:, 6])
    a7 = stat.median(result[:, 7])
    a8 = stat.median(result[:, 8])
    a9 = stat.median(result[:, 9])
    a10 = stat.median(result[:, 10])
    a11 = stat.median(result[:, 11])
    a12 = stat.median(result[:, 12])

    print('{:.0f}\t {:.0f}\t {:.0f}\t {:.0f}\t {:.0f}\t {:.0f}\t {:.7f}\t {:.7f}\t {:.7f}\t {:.7f}\t {:.7f}\t {:.7f}\t {:.7f}'.
        format(a0,     a1,       a2,     a3,      a4,     a5,      a6,      a7,       a8,      a9,      a10,     a11,    a12))

def calc_quality_end(matrix, filename):
    def calc_mistakes(matrix): #  расчет кол-ва одинаковых ответов в матрице при условии разных данных на вход
        new_matrix = sorted(matrix, key=lambda data: data[0])
        new_matrix = sorted(new_matrix, key=lambda data: data[1])
        new_matrix = sorted(new_matrix, key=lambda data: data[2])
        new_matrix = sorted(new_matrix, key=lambda data: data[3])
        new_matrix = np.array(new_matrix)
        r,c = new_matrix.shape
        cnt = 0
        label = new_matrix[0, 0]
        tmp1 = new_matrix[0, 1]
        tmp2 = new_matrix[0, 2]
        tmp3 = new_matrix[0, 3]
        for i in range (1,r):
            if ((label != new_matrix[i,0]) &
                (new_matrix[i, 1] == tmp1) & (new_matrix[i,2] == tmp2) & (new_matrix[i,3] == tmp3)):
                cnt += 1
            else:
                label = new_matrix[i, 0]
                tmp1 = new_matrix[i, 1]
                tmp2 = new_matrix[i, 2]
                tmp3 = new_matrix[i, 3]
        #print("cnt " + str(cnt))
        return cnt

    def calc_distance(matrix):
        ones = matrix[matrix[:, 0] == 1]
        zeros = matrix[matrix[:, 0] == 0]
        cgx_o, cgy_o, cgz_o = calc_gravity_center(ones)
        cgx_z, cgy_z, cgz_z = calc_gravity_center(zeros)
        #print("distance" , math.sqrt((cgx_o-cgx_z)**2 + (cgy_o-cgy_z)**2 + (cgz_o - cgz_z)**2 ))
        dist = math.sqrt((cgx_o - cgx_z) ** 2 + (cgy_o - cgy_z) ** 2 + (cgz_o - cgz_z) ** 2)
        radiusO = calc_radius (cgx_o, cgy_o, cgz_o, ones)
        radiusZ = calc_radius(cgx_z, cgy_z, cgz_z, zeros)
        return dist, radiusO, radiusZ

    def calc_radius(x,y,z,matrix):
        r, c = matrix.shape
        radius = 0
        for i in range(r):
            radius += math.sqrt((matrix[i,1] - x)**2 + (matrix[i,2] - y)**2 +(matrix[i,3] - z)**2 )
        radius /= r
        return radius

    def calc_gravity_center(matrix):
        r, c = matrix.shape
        mass = np.array(np.ones(r))
        cgx = np.sum(matrix[:, 1] * mass) / np.sum(mass)
        cgy = np.sum(matrix[:, 2] * mass) / np.sum(mass)
        cgz = np.sum(matrix[:, 3] * mass) / np.sum(mass)
        return (cgx, cgy, cgz)

    # 0 - Label
    # 1,2,3 - DecisionNeurons
    # 4 - AnswerLength
    # 5 - Tick
    # 6 - velocity
    # 7 - start of new data


    # рассматриваем только с конца: последние  10, 15, 20, 25, 50 % датасета
    r,c = matrix.shape
    starts =np.zeros(5)
    starts[0] = math.floor(90 * r / 100)
    starts[1] = math.floor(85 * r / 100)
    starts[2] = math.floor(80 * r / 100)
    starts[3] = math.floor(75 * r / 100)
    starts[4] = math.floor(50 * r / 100)

    result = np.zeros(shape=(5, 13))

    for i in range(0,5):
        start = starts[i]
        finish = r-1
        result[i, 0] = matrix[start, 7] # time
        result[i, 1] = stat.median(matrix[start:finish, 4]) # average_answer_length
        result[i, 2] = stat.stdev(matrix[start:finish, 4])
        result[i, 3] = stat.median(matrix[start:finish, 6])  # average_velocity
        result[i, 4] = stat.stdev(matrix[start:finish, 6])
        result[i, 5] = calc_mistakes(matrix[start:finish, 0:4]) # number of mistakes
        dist, radiusO, radiusZ = calc_distance(matrix[start:finish, 0:4])
        result[i, 7] = dist # average distance between center of mass
        result[i, 9] = radiusO # average radius of ONES from center of mass
        result[i, 11] = radiusZ  # average radius of ZEROS from center of mass

    for i in range(0,5):
        result[i, 6] = result[i,5] / max(result[:,5]) # NORMED number of mistakes
        result[i, 8] = result[i,7] / max(result[:,7]) # NORMED average distance between center of mass
        result[i, 10] = result[i, 9] / max(result[:, 9])  # NORMED average radius of ONES from center of mass
        result[i, 12] = result[i, 11] / max(result[:, 11])  # NORMED average radius of ZEROS from center of mass

    for i in range(0, 5):
        print('{} \t {:.0f}\t {:.0f}\t {:.0f}\t {:.0f}\t {:.0f}\t {:.0f}\t {:.7f}\t {:.7f}\t {:.7f}\t {:.7f}\t {:.7f}\t '
              '{:.7f}\t {:.7f}'.format(filename, result[i,0],result[i,1],result[i,2],result[i,3],result[i,4],result[i,5],
                                result[i,6],result[i,7],result[i,8],result[i,9],result[i,10],result[i,11],result[i,12]))


def calc_statistic(matrix, filename):
    # 0 - Label
    # 1,2,3 - DecisionNeurons
    # 4 - AnswerLength
    # 5 - Tick
    # 6 - velocity
    # 7 - start of new data
    r, c = matrix.shape

    set_size = math.floor(r / 10) - 1
    step = math.floor(r / 20) - 1
    result = np.zeros(shape=(19, 11))

    for i in range(0, 19):
        start = i * step
        finish = i * step + set_size
        result[i, 0] = matrix[start, 7]  # average_answer_length
        result[i, 1] = stat.median(matrix[start:finish, 4])  # average_answer_length
        result[i, 2] = stat.stdev(matrix[start:finish, 4])
        result[i, 3] = stat.median(matrix[start:finish, 6])  # average_velocity
        result[i, 4] = stat.stdev(matrix[start:finish, 6])
        result[i, 5] = stat.median(matrix[start:finish, 1])  # average_dec_neuron_#1
        result[i, 6] = stat.stdev(matrix[start:finish, 1])
        result[i, 7] = stat.median(matrix[start:finish, 2])  # average_dec_neuron_#2
        result[i, 8] = stat.stdev(matrix[start:finish, 2])
        result[i, 9] = stat.median(matrix[start:finish, 3])  # average_dec_neuron_#3
        result[i, 10] = stat.stdev(matrix[start:finish, 3])
    write_data_statistic(result, 19, filename + '_stat')

def move_to_plane(matrix):
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
    fun = functools.partial(error, points=matrix[:, 1:4])
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)

    print("basis plane found")

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

    newMatrix = np.array(
        [[point[0], np.dot(point[1:4], xbasis), np.dot(point[1:4], ybasis), np.dot(point[1:4], norm) - c / normLen,
          point[4], point[5], point[6], point[7]] for point in matrix])

    print("moved to new coords")

    # находим центр масс для перехода в полярные координаты
    # первая интерация: все массы одинаковы
    # вторая итерация: масса обратно пропорциональа растоянию до ц.м.№1
    # последняя итерация: находим точку, ближайшую к ц.м.№1 и ц.м.№2
    r, c = newMatrix.shape
    massX = np.array(np.ones(r))
    massY = np.array(np.ones(r))
    cgx1 = np.sum(newMatrix[:, 1] * massX) / np.sum(massX)
    cgy1 = np.sum(newMatrix[:, 2] * massY) / np.sum(massY)
    for i in range(r):
        massX[i] = 1 / abs(newMatrix[i, 1] - cgx1)
        massY[i] = 1 / abs(newMatrix[i, 2] - cgy1)
    cgx2 = np.sum(newMatrix[:, 1] * massX) / np.sum(massX)
    cgy2 = np.sum(newMatrix[:, 2] * massY) / np.sum(massY)

    minDist = 1000
    for i in range(r):
        dist = (newMatrix[i, 1] - cgx1) ** 2 + (newMatrix[i, 2] - cgy1) ** 2
        dist += (newMatrix[i, 1] - cgx2) ** 2 + (newMatrix[i, 2] - cgy2) ** 2
        if (dist < minDist):
            cgx = newMatrix[i, 1]
            cgy = newMatrix[i, 2]
            minDist = dist

    print("mass center found")

    # переход в полярные координаты
    newMatrix = np.array([[point[0], point[1] - cgx, point[2] - cgy, point[3],
                           point[4], point[5], point[6], point[7]] for point in newMatrix])

    print("moved to mass center #1")
    newMatrix = np.array(
        [[point[0], math.sqrt(point[1] ** 2 + point[2] ** 2), math.atan(point[2] / point[1]), point[3],
          point[4], point[5], point[6], point[7]] for point in
         newMatrix if point[1] != 0 and point[2] != 0])
    print("moved to mass center #2")
    # сохранение файла с данными
    # filename = filename[:-4] + '_mod3.dat'
    # getMatrix(newMatrix, filename)
    return newMatrix

#def plotData(x,y):
    #plt.plot(x, y, linewidth=2.0, color='blue')
    #plt.xlabel('some numbers')
    #plt.ylabel('some numbers')
    #plt.axis([0, 6, 0, 20])
    #plt.show()

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
        (r,c) = matrix.shape
        wr = csv.writer(csvfile, delimiter=',')
        for i in range(r):
            wr.writerow(matrix[i,:])
    return 0

def write_data_statistic(matrix, n, filename):
    with open(filename, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow(['time', 'av.answer.length', 'sd.answer.length', 'av.velocity', 'sd.velocity', 'av.neur250',
                     'sd.neur250', 'av.neur251', 'sd.neur251', 'av.neur252', 'sd.neur252'])
        for i in range(n):
            wr.writerow(matrix[i,:])
    return 0

def write_data_quality(matrix, n, filename):
    with open(filename, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow(['time', 'av.answer.length', 'sd.answer.length', 'av.velocity', 'sd.velocity',
                     'mistakes', 'mistakesNORM', 'centerGravityDistance', 'centerGravityDistanceNORM',
                     'averRadOnes', 'averRadOnesNorm', 'averRadZeros', 'averRadZerosNorm'])
        for i in range(n):
            wr.writerow(matrix[i,:])
    return 0

############################################################
#GET ONE DATASET
#X = getData('1486416673294.csv', '1486416673294')

# GET SOME DATA
# filenames =  ('1486416673294.csv', '1486430023690.csv', '1486435241183.csv', '1486444008135.csv', '1486448918687.csv')
# for f in filenames:
#   getData(f, os.path.split(f)[1])
#   print ("\n------------------------------------\n")

# GET ALL DATA
with tarfile.open('practice_dataset.tar.gz', 'r:gz') as tar:
    for member in tar:
        if os.path.splitext(member.name)[1] != '.csv':
            continue
        getData(tar.extractfile(member), os.path.split(member.name)[1])


print("--->>>Done!<<<---")

