import sympy
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

sympy.init_printing()

'''
    Required packages to install before running the script:
        - sympy
        - scipy
        - numpy
        - matplotlib
        
    Installing instruction:
        - open cmd
        - run command for all following packages: pip install package_name
'''

x = sympy.Symbol('x')


class Integral:

    def __init__(self):
        self._integral_formula = ''
        self.integral_formula_result = ''
        self._boundary_from = 0
        self._boundary_to = 0
        self.step = 0.01


    def get_integral_formula(self):
        self.integral_formula_str = input('Enter integral formula with "x" symbol: ')
        boundary_from = input('Enter low boundary as number: ')
        boundary_to = input('Enter high boundary as number: ')

        self._integral_formula = eval(self.integral_formula_str)
        self._boundary_from = int(boundary_from)
        self._boundary_to = int(boundary_to)


    def calculate_integral_formula(self):
        self.integral_formula_result = sympy.integrate(self._integral_formula, x)


    def prepare_integral_formula(self, xx):
        changed_formula = self.integral_formula_str.replace('x', 'xx')
        return eval(changed_formula)


    def calculate_integral_quad(self):
        how_many_quads = int((self._boundary_to - self._boundary_from) / self.step)
        #self.quad_result = scipy.integrate.quad(self.prepare_integral_formula, self._boundary_from, self._boundary_to)
        self.quad_result = scipy.integrate.fixed_quad(self.prepare_integral_formula,
                                                      self._boundary_from,
                                                      self._boundary_to,
                                                      n=how_many_quads)


    def calculate_trapz(self, x_line: np.ndarray):
        fun_res = self.prepare_integral_formula(x_line)
        return np.trapz(fun_res, x_line)


    def calculate_simps(self, x_line: np.ndarray):
        fun_res = self.prepare_integral_formula(x_line)
        return scipy.integrate.simps(fun_res, x_line)


    def calculate_integral_trapz(self):
        x_line = np.arange(self._boundary_from, self._boundary_to + self.step, self.step)
        self.trapz_result = self.calculate_trapz(x_line)


    def calculate_integral_simps(self):
        x_line = np.arange(self._boundary_from, self._boundary_to + self.step, self.step)
        self.simps_result = self.calculate_simps(x_line)


    def print_results(self):
        print(15*'*', 'results', 15*'*')
        print('Inserted formula:', self.integral_formula_str)
        print('Formula result:', self.integral_formula_result)
        print('Step:', self.step)
        print(f'Numeric result - Quad method: {self.quad_result[0]:.4f}')
        print(f'Numeric result - Trapz method: {self.trapz_result:.4f}')
        print(f'Numeric result - Simps method: {self.simps_result:.4f}')
        print(40 * '*')


    def prepare_plot_data(self):
        x_line = np.arange(self._boundary_from, self._boundary_to + self.step, self.step)
        #fun_val = [self.prepare_integral_formula(xx) for xx in x_line]
        fun_val = self.prepare_integral_formula(x_line)
        self.preapre_plot(x_line, fun_val)


    def preapre_plot(self, xx, yy):
        figure = plt.figure()
        plt.plot(xx, yy)
        plt.grid(True)
        plt.legend([f'Function: {self.integral_formula_str}'])
        plt.fill_between(x=xx,
                         y1=yy,
                         where=(xx>xx[0]) & (xx<xx[-1]),
                         color="b",
                         alpha=0.2)
        plt.show()


if __name__ == '__main__':

    integral = Integral()
    integral.get_integral_formula()
    integral.calculate_integral_formula()
    integral.calculate_integral_quad()
    integral.calculate_integral_trapz()
    integral.calculate_integral_simps()
    integral.print_results()
    integral.prepare_plot_data()

