from matplotlib import pyplot as plt
import numpy as np

'''
    Calculation of interpolation polynominal with Lagrange method.
'''


def calculate_fun_values(x_vector, dagree_of_polynominal, x_points, y_points):

    yp_vector = []

    for xp in x_vector:
        yp = 0
        for i in range(dagree_of_polynominal+1):
            p = 1
            for j in range(dagree_of_polynominal+1):
                if j != i:
                    p *= (xp - x_points[j]) / (x_points[i] - x_points[j])

            yp += y_points[i] * p

        yp_vector.append(yp)

    return yp_vector


def prepare_plot(x_fun_vector, y_fun_vector, x_points, y_points):

    figure = plt.figure()
    plt.plot(x_points, y_points, 'ro', x_fun_vector, y_fun_vector, 'b-')
    plt.grid(True)
    plt.legend(['Points to interpolate', 'Interpolation polynominal'])
    plt.show()


if __name__ == '__main__':

    x_points = input('Enter x positions in list, e.g.: [0,20,40,60,80,100]: ')
    x_points = eval(x_points)
    y_points = input('Enter y positions in list, e.g.: [40,50,60,30,90,100]: ')
    y_points = eval(y_points)

    dagree_of_polynominal = len(x_points) - 1

    print(f'X positions: {x_points}')
    print(f'X positions: {y_points}')
    print(f'Dagree of polynominal: {dagree_of_polynominal}')

    x_fun_len = 1000
    x_fun_vector = np.linspace(x_points[0], x_points[-1], num=x_fun_len)

    y_fun_vector = calculate_fun_values(x_fun_vector, dagree_of_polynominal, x_points, y_points)

    prepare_plot(x_fun_vector, y_fun_vector, x_points, y_points)



