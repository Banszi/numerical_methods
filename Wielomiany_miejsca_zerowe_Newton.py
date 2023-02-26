import sympy

'''
    Libraries that are needed to be installed: sympy
    
    CMD command: pip install sympy
'''

# Define x as symbol
x = sympy.Symbol('x')


def calculate_result_from_function(fun, x_guess):
    function_converted_to_str = str(fun)
    function_converted_to_str = function_converted_to_str.replace('x', 'x_guess')
    f = eval(function_converted_to_str)
    return f


def calculate_result_from_function_derivative(fun_derivative, x_guess):
    function_deriv_converted_to_str = str(fun_derivative)
    function_deriv_converted_to_str = function_deriv_converted_to_str.replace('x', 'x_guess')
    df = eval(function_deriv_converted_to_str)
    return df


def define_function_derivative(function_formula):
    return function_formula.diff(x)


def newton_method(fun, fun_deriv, x_guess, how_many_iterations):
    for iteration in range(1, how_many_iterations):
        f = calculate_result_from_function(fun, x_guess)
        df = calculate_result_from_function_derivative(fun_deriv, x_guess)

        i = x_guess - (f / df)
        x_guess = i

    res_x = x_guess

    return res_x, how_many_iterations


if __name__ == '__main__':

    print('If you will use power of any number then write it as pow(x,y)')
    print('Example 1: x**2 -> pow(x,2)')
    print('Example 2: x**2-2 -> pow(x,2)-2')
    function_str = input('Insert function with parameter as x: ')
    x_guess = int(input('Insert first guess of x as int: '))
    how_namy_iterations = int(input('Insert how many iterations program should calculate: '))
    function_formula = eval(function_str)
    print(f'Function formula: {function_formula}')
    function_derivative_formula = define_function_derivative(function_formula)
    print(f'Derivative of function: {function_derivative_formula}')
    result_x, iterations = newton_method(function_formula, function_derivative_formula, x_guess, how_namy_iterations)
    print(f'\nAfter {iterations} iterations found root is equal to: {result_x}')
