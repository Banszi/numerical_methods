import sys


def calculate_result_from_function(fun_formula, x):
    f = eval(fun_formula)
    return f


def newton_method(fun_formula, a_boundary, b_boundary, accept_error):
    i = 0
    c_before = 0
    f_a = calculate_result_from_function(fun_formula, a_boundary)
    f_b = calculate_result_from_function(fun_formula, b_boundary)
    c = (a_boundary * f_b - b_boundary * f_a) / (f_b - f_a)
    error = abs(c - c_before)

    while error > accept_error:

        c_after = (a_boundary * f_b - b_boundary * f_a) / (f_b - f_a)
        f_a = calculate_result_from_function(fun_formula, a_boundary)
        f_b = calculate_result_from_function(fun_formula, b_boundary)
        f_c = calculate_result_from_function(fun_formula, c_after)

        if f_a * f_b >= 0:
            print('No root present in given boudary numbers!')
            sys.exit()
        elif f_c * f_a < 0 :
            error = abs(c_after - b_boundary)
            b_boundary = c_after
            i += 1
        elif f_c * f_b < 0:
            error = abs(c_after - a_boundary)
            a_boundary = c_after
            i += 1
        else:
            print('Example can not be calculated. Change paramerets!')
            sys.exit()

    return error, i, c_after


if __name__ == '__main__':

    function_str = input('Insert function with parameter as x: ')
    a_boundary = int(input('Insert a boudary as int: '))
    b_boundary = int(input('Insert b boudary as int: '))
    accept_error = float(input('Insert acceptance error as float (e.g. 0.001): '))
    print(f'Function formula: {function_str}')
    error, i, root = newton_method(function_str, a_boundary, b_boundary, accept_error)

    print(f'After {i} iterations error is queal to: {error:.4f}')
    print(f'Found root is equal to: {root:.4f}')


