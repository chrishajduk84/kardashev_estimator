from abc import ABC
import numpy as np

class BaseInterpolator(ABC):
    def __init__(self):
        pass

    def generate_polynomial(self, data):
        """
        Generates a set of newton polynomials coefficients
        :param data: Data provided in (x, y) tuple format
        :return: list containing ordered polynomial coefficients
        """


class NewtonPolynomialInterpolator(BaseInterpolator):
    """ This will generate an interpolation object which can continuously be updated with new data (ex: live data)

        Provide initial data -> generate polynomials -> create interpolation function
        Extend dataset -> generate new polynomials only -> update interpolation function
        Update dataset -> re-generate all affected polynomials (everything after) -> update interpolation function
        Delete from dataset -> re-generate all affected polynomials (everything after) -> update interpolation function
    """


    def __init__(self, initial_data):
        """
        This will use the initial data provided to generate the interpolation
        :param initial_data: List of (x,y) tuples to generate the Newton interpolation from
        """
        self.__generate_coefficients(initial_data)

    def __generate_coefficients(self, data):
        # Tuple definitions for readability
        X = 0
        Y = 1
        n = len(data)

        # Initialize storage for coefficients
        self.coefficient_matrix = np.zeros([n, n])
        for i in range(n):
            self.coefficient_matrix[i, 0] = data[i][Y]

        for j in range(1, n):
            for i in range(n - j):
                self.coefficient_matrix[i][j] = (self.coefficient_matrix[i+1][j-1] - self.coefficient_matrix[i][j-1]) \
                                                / (data[i+j][X] - data[i][X])

        # Save x_data and y_data for later
        self.x_data = [x for x, y in data]
        self.y_data = [y for x, y in data]

        # What data do we need to store to be able to update coefficents in the future????
        # If we have too much data, we may want to figure out how to reduce space and time requirements

    @property
    def coefficients(self):
        return self.coefficient_matrix[0, :]

    def predict(self, unix_time):
        coeff = self.coefficients
        n = len(coeff) - 1
        print(n)
        # TODO: Data here doesn't seem to make sense.... why?
        # How is AFG producing 4443303248615.493 TW of power?
        # AFG - 4443303248615.493
        # ALB - -1168411916096.3054
        # DZA - -102149948625961.67
        # ASM - 0.0

        val = coeff[n]
        for i in range(1, len(coeff)):
            val = coeff[n-i] + (unix_time - self.x_data[n - i])*val
        return val
