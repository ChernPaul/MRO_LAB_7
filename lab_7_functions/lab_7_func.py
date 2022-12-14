import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show

NUMBER_OF_VECTOR_DIMENSIONS = 2
SAMPLE_SIZE_N = 200

X_LOWER_BORDER = -3
X_UPPER_BORDER = 2
Y_LOWER_BORDER = -3
Y_UPPER_BORDER = 2

FIGSIZE_PARAMETER_1 = 10
FIGSIZE_PARAMETER_2 = 5
# from lab 1
def generate_vector_X(A, M, n, N):
    left_border = 0
    right_border = 1
    m = (right_border + left_border) / 2
    number_of_realizations = 50
    Sn = np.zeros((n, N))
    for i in range(0, number_of_realizations, 1):
        Sn += np.random.uniform(left_border, right_border, (n, N)) - m
    standard_deviation = (right_border - left_border) / np.sqrt(12)
    E = Sn / (standard_deviation * np.sqrt(number_of_realizations))
    X = np.matmul(A, E) + np.reshape(M, (2, 1)) * np.ones((1, N))
    return X


def calculate_matrix_A(B):
    matrix_A = np.zeros((2, 2))
    matrix_A[0][0] = np.sqrt(B[0][0])
    matrix_A[0][1] = 0
    matrix_A[1][0] = B[0][1] / np.sqrt(B[0][0])
    matrix_A[1][1] = np.sqrt(B[1][1] - (B[0][1] ** 2) / B[0][0])
    return matrix_A


def calculate_mathematical_expectation_M(x):
    M = np.sum(x, axis=1) / SAMPLE_SIZE_N
    return M


def get_B_correlation_matrix_for_vector(x):
    M = calculate_mathematical_expectation_M(x)
    # M shape is (1, 2)
    B = np.zeros((2, 2))
    for i in range(0, SAMPLE_SIZE_N, 1):
        # sum for i xi * xi ^t  where x[:, i] = [ x, y ]^t  shape = (number of columns, number of rows) x.shape = (1,2)
        tmp = np.reshape(x[:, i], (2, 1))
        B += (np.matmul(tmp, np.transpose(tmp)))
    B /= SAMPLE_SIZE_N
    B -= np.matmul(np.reshape(M, (2, 1)), np.transpose(np.reshape(M, (2, 1))))
    return B



# for current lab

def d(x, z):
    dist = np.sum(np.square(x-z))
    return np.sqrt(dist)

Distance = np.vectorize(d, signature='(n),(m)->()')

def maxminMethod(vectors_as_one_vector):
    result = copy.copy(vectors_as_one_vector)
    clusters = []
    arrM = []
    # step 1 started
    M_all = vectors_as_one_vector.sum(axis=1) / len(vectors_as_one_vector[0])
    result = np.transpose(result)
    # вектор удаление которого от среднего всех векторов выборки максимально
    distances = Distance(result, M_all)
    m0 = result[np.argmax(distances)]
    clusters.append([m0])
    arrM.append(m0)
    result = np.delete(result, np.argmax(distances), axis=0)
    # step 1 finished
    # step 2 started
    distances = Distance(result, m0)
    m1 = result[np.argmax(distances)]
    arrM.append(m1)
    clusters.append([m1])
    result = np.delete(result, np.argmax(distances), axis=0)
    # step 2 finished
    dtypical = [Distance(m0, m1) / 2]
    dmin = [dtypical[-1] + 1]
    legends = ["M(x)", "class 0", "class 1"]


    # distanceTable
    #             x(0)          x(1)        ...       x(i)        ...       x(N-1)
    # M(0)      d(M0,x0)     d(M0, x1)      ...     d(M0, xi)     ...    d(M0, x(N-1))
    # M(1)      d(M1,x0)     d(M1, x1)      ...     d(M1, xi)     ...    d(M1, x(N-1))
    # ...         ...           ...         ...       ...         ...        ...
    # M(i)      d(Mi,x0)     d(Mi, x1)      ...     d(Mi, xi)     ...    d(Mi, x(N-1))
    # ...         ...           ...         ...       ...         ...        ...
    # M(L-2)  d(M(L-2),x0)  d(M(L-2), x1)   ...   d(M(L-2), xi)   ...   d(M(L-2), x(N-1))
    number_of_clusters = 2
    while dmin[-1] > dtypical[-1]:
        distanceTable = []
        for i in range(0, len(arrM)):
            distanceTable.append(Distance(result, arrM[i]))

        # distribution by existing clusters
        l = np.argmin(np.transpose(distanceTable), axis=1)
        tmp = copy.deepcopy(clusters)
        for k in range(0, len(result)):
            tmp[l[k]].append(result[k])
        # output
        fig0 = plt.figure(figsize=(10, 10))
        viewData = []
        for k in range(0, len(tmp)):
            viewData.append(np.transpose(tmp[k]))
        tmp_ = np.transpose(arrM)
        plt.suptitle('Clusters positions for number of clusters equal:  ' + str(number_of_clusters))
        plt.xlim(X_LOWER_BORDER, X_UPPER_BORDER)
        plt.ylim(Y_LOWER_BORDER, Y_UPPER_BORDER)
        plt.plot(tmp_[0], tmp_[1], 'ko')
        c = ['r+', 'b+', 'g+', 'y+', 'c+', 'm+']
        for i in range(0, len(viewData)):
            plt.plot(viewData[i][0], viewData[i][1], c[i % len(c)], label='cluster_' + str(i + 1))
        plt.legend()

        # создание нового кластера(если надо)
        minDistances = np.min(np.transpose(distanceTable), axis=1)
        M_ = result[np.argmax(minDistances)]
        dmin.append(np.min(Distance(arrM, M_)))
        if dmin[-1] > dtypical[-1]:
            legends.append(f"class {len(arrM)}")
            arrM.append(M_)
            clusters.append([M_])
            result = np.delete(result, np.argmax(minDistances), axis=0)
            dtypical.append(0)
            for j in range(0, len(arrM)):
                dtypical[-1] += np.sum(Distance(arrM, arrM[j]))
            dtypical[-1] /= 2*len(arrM)*(len(arrM) - 1)
        number_of_clusters += 1
    dmin.pop(0)

    for k in range(0, len(result)):
        clusters[l[k]].append(result[k])
    return clusters, dmin, dtypical, arrM


def viewClusters(datas, arrM, fig, subplt_index, title=''):
    viewData = []
    for k in range(0, len(datas)):
        viewData.append(np.transpose(datas[k]))
    tmp = np.transpose(arrM)

    fig.add_subplot(subplt_index)
    plt.title('Clusters positions for ' + title)
    plt.xlim(X_LOWER_BORDER, X_UPPER_BORDER)
    plt.ylim(Y_LOWER_BORDER, Y_UPPER_BORDER)
    plt.plot(tmp[0], tmp[1], 'ko')
    c = ['r+', 'b+', 'g+', 'y+', 'c+', 'm+']
    for i in range(0, len(viewData)):
        plt.plot(viewData[i][0], viewData[i][1], c[i % len(c)], label='cluster_' + str(i+1))
    plt.legend()
    return fig

def K_introGroupAvg(vectors, initVectors):
    K = len(np.transpose(initVectors))
    clusters = []
    legends = ["M(x)"]
    for i in range(0, K):
        clusters.append([])
        legends.append(f"class {i}")

    result = np.transpose(vectors)
    new_arrM = copy.deepcopy(np.transpose(initVectors))
    prev_arrM = np.mean(result, axis=0)*np.ones_like(new_arrM)
    prev_k = np.zeros((len(result),)).astype(int)
    imposters = [-1]

    # fig0 = plt.figure(figsize=(10, 10))
    tmp = [result]
    while not (imposters[-1] == 0):  # not ((imposters[-1] == 0) | (new_arrM == prev_arrM).all())
        distances = []
        for i in range(0, K):
            distances.append(Distance(result, new_arrM[i]))
        # classes affiliation
        new_k = np.argmin(distances, axis=0)

        # classes splitting for vcetors
        copyClusters = copy.deepcopy(clusters)
        for i in range(0, len(result)):
            copyClusters[new_k[i]].append(result[i])

        tmp = copy.copy(copyClusters)

        # number of vectors that changed class
        imposters.append(list(new_k == prev_k).count(False))
        prev_k = copy.copy(new_k)

        # ME updating
        prev_arrM = copy.copy(new_arrM)
        for i in range(0, K):
            new_arrM[i] = np.mean(copyClusters[i], axis=0)
    imposters.pop(0)

    show()
    for j in range(0, len(result)):
        clusters[new_k[j]].append(result[j])
    return clusters, new_arrM, imposters


def Imposters(ay):
    fig = plt.figure(figsize=(7, 7))
    ax = np.arange(0, len(ay), 1)
    plt.plot(ax, ay, 'b-')
    plt.legend(["Number of changed class from iteration"])


def simpleView(vectors, classes, Ms, title=''):
    colors = ['r+', 'g+', 'b+', 'y+', 'c+', '+m']
    fig = plt.figure(figsize=(FIGSIZE_PARAMETER_1, FIGSIZE_PARAMETER_2))
    plt.suptitle(title)
    fig.add_subplot(1, 2, 1)
    plt.title('Source vectors')
    plt.xlim(X_LOWER_BORDER, X_UPPER_BORDER)
    plt.ylim(Y_LOWER_BORDER, Y_UPPER_BORDER)

    plt.plot(vectors[0][0], vectors[0][1], 'r+', label='vector_1')
    plt.plot(vectors[1][0], vectors[1][1], 'g+', label='vector_2')
    plt.plot(vectors[2][0], vectors[2][1], 'b+', label='vector_3')
    plt.plot(vectors[3][0], vectors[3][1], 'y+', label='vector_4')
    plt.plot(vectors[4][0], vectors[4][1], 'c+', label='vector_5')
    plt.legend()
    viewClusters(classes, Ms, fig, 122, 'Algorithm result')
    show()