import numpy as np
import matplotlib.pyplot as plt
from skimage.io import show

from lab_7_functions.lab_7_func import calculate_matrix_A, generate_vector_X, maxminMethod,  \
    Imposters, K_introGroupAvg, simpleView

TASK_VARIANT = 20

M1 = [1, -1]
M2 = [-2, -2]
M3 = [-1, 1]
M4 = [0, 0]
M5 = [1, 1]

B1 = [[0.05, 0.001],
      [0.001, 0.05]]

B2 = [[0.1, 0.02],
      [0.02, 0.1]]

B3 = [[0.1, 0.05],
      [0.05, 0.1]]

B4 = [[0.05, 0.005],
      [0.005, 0.05]]

B5 = [[0.08, -0.02],
      [-0.02, 0.08]]

PROBABILITY_HALF_OF_ONE = 0.5
PROBABILITY_ONE_OF_TREE = float(1/3)

C_MATRIX_OF_FINE = np.array([[0.0, 1.0],
                             [1.0, 0.0]])


X_LOWER_BORDER = -3
X_UPPER_BORDER = 2
Y_LOWER_BORDER = -3
Y_UPPER_BORDER = 2

FIGSIZE_PARAMETER_1 = 10
FIGSIZE_PARAMETER_2 = 5

NUMBER_OF_VECTOR_DIMENSIONS = 2
SAMPLE_SIZE_N = 50

if __name__ == '__main__':

    # 1. Смоделировать и изобрать графически обучающие выборки объема =50 для пяти нормально распределенных
    # случайных величин с заданными МО и самотоятельно подобранными корреляционными матрицами
    # которые обеспечивают линейную разделимость классов
    """
    A1 = calculate_matrix_A(B1)
    A2 = calculate_matrix_A(B2)
    A3 = calculate_matrix_A(B3)
    A4 = calculate_matrix_A(B4)
    A5 = calculate_matrix_A(B5)

    vector_1 = generate_vector_X(A1, M1, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_2 = generate_vector_X(A2, M2, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_3 = generate_vector_X(A3, M3, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_4 = generate_vector_X(A4, M4, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_5 = generate_vector_X(A5, M5, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    """
    # np.save("saves\\vector_1", vector_1)
    # np.save("saves\\vector_2", vector_2)
    # np.save("saves\\vector_3", vector_3)
    # np.save("saves\\vector_4", vector_4)
    # np.save("saves\\vector_5", vector_5)


    vector_1 = np.load("saves\\vector_1.npy")
    vector_2 = np.load("saves\\vector_2.npy")
    vector_3 = np.load("saves\\vector_3.npy")
    vector_4 = np.load("saves\\vector_4.npy")
    vector_5 = np.load("saves\\vector_5.npy")

    vectors = [vector_1, vector_2, vector_3, vector_4, vector_5]

    fig_new = plt.figure(figsize=(10, 10))
    plt.title('Source data')
    plt.xlim(X_LOWER_BORDER, X_UPPER_BORDER)
    plt.ylim(Y_LOWER_BORDER, Y_UPPER_BORDER)
    plt.plot(vector_1[0], vector_1[1], 'r+', label='vector_1')
    plt.plot(vector_2[0], vector_2[1], 'g+', label='vector_2')
    plt.plot(vector_3[0], vector_3[1], 'b+', label='vector_3')
    plt.plot(vector_4[0], vector_4[1], 'y+', label='vector_4')
    plt.plot(vector_5[0], vector_5[1], 'c+', label='vector_5')
    plt.legend()
    show()


    # 2. Объединить пять выборок в одну. Общее количество векторов в выборке должно быть равным 250.
    # Полученная объединенная выборка используется для выполнения пунктов 3 и 4 настоящего плана.

    common_selection = np.concatenate((vector_1, vector_2, vector_3, vector_4, vector_5), axis=1)



    # 3. Разработать программу кластеризации данных с использованием минимаксного алгоритма.
    # В качестве типичного расстояния взять половину среденего расстояния между существующими кластерами
    # Построить отображение результатов кластеризации для числа кластеров начиная с двух. Построить график
    # зависимости максимального (из минимальных) и типичного расстояний от числа кластеров.


    classes, d_min, d_typical, arr_M = maxminMethod(common_selection)
    simpleView(vectors, classes, arr_M, 'Min Max Algorithm ')

    # building graphic of dependency d_max_from_min and d_typical from clusters quantity
    x = np.arange(2, 2+len(d_min), 1)
    fig = plt.figure(figsize=(FIGSIZE_PARAMETER_1, FIGSIZE_PARAMETER_2))
    plt.title('Graphic dependency d_max and d_typical from number of clusters')
    plt.plot(x, d_min, c="b", marker="o", linestyle="-")
    plt.plot(x, d_typical, c="r", marker="o", linestyle="-")
    plt.legend(["d min", "d typical"])
    show()


    # 4. Разработать программу кластеризации данных с использованием алгоритма К внутригрупповых средних
    # для числа кластеров равного 3 и 5. Для ситуации 5 кластеров подобрать начальные условия так, чтобы получить
    # 2 результата а) чтобы кластеризация максимально соответствовала первоначальному разбиению на классы
    # (правльная кластеризация) б) чтобы кластеризация максимально не соответсвовала первоначальному разбиению на классы
    # (неправильная кластеризация) Для всех случаев построить графики зависимости числа векторов признаков,
    # сменивших номер кластера от номера итерации алгоритма.

    classes3, M3, imposter3 = K_introGroupAvg(common_selection, common_selection[:, [151, 156, 160]])
    simpleView(vectors, classes3, M3, 'Number of clusters = 3')
    Imposters(imposter3)
    show()

    classes5, M5, imposters5 = K_introGroupAvg(common_selection, common_selection[:, [151, 156, 170, 175, 166]])
    simpleView(vectors, classes5, M5, 'Number of clusters = 5 Correct classificaton')
    Imposters(imposters5)
    show()

    fakeClasses5, fakeM5, fakeImposters5 = K_introGroupAvg(common_selection, common_selection[:, [51, 52, 53, 64, 55]])
    simpleView(vectors, fakeClasses5, fakeM5, 'Number of clusters = 5 wrong classificaton')
    Imposters(fakeImposters5)
    show()


