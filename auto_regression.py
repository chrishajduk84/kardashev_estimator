from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def differenced_series(data, interval=1):
    diff = []
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return diff


class PowerRegression:
    
    def __init__(self, data):
        """ The initialization of this class must contain a list of (x,y) tuple data. This class will ensure the data is linearly interpolated to create a constant periodicity. """

        self.data = data
        self.data_step = self.__min_data_step(self.data)
        # Are any data steps missing? Do we need to interpolate for those?
        interpolated_data = self.__interpolate_data_steps(self.data, self.data_step)

        # Generate linear differences list
        self.diffs = []
        # ARIMA model uses 3 parameters (in addition to the data)
        # p = the number of data points to complete the regression on
        # d = integration order (d=1 means f(t) - f(t-1), d=2 would use double that difference)
        # q = the averaging window
        # order = (p,d,q)
        self.model = ARIMA(self.diffs, order=(len(data), 0, len(data) - 1))

    def interpolate(self):
        pass

    def __min_data_step(self, data):
        """ Using the data provided in the initialization, 
        this function will determine the step size used for interpolation (if necessary) """

        if len(data) < 2:
            return None

        step_size = data[1][0] - data[0][0]
        for i in range(2, len(data)):
            step_size = min(step_size, data[i][0] - data[i-1][0])
        return step_size

    def __interpolate_data_steps(self, data, step_size):
        first_step = data[0][0]
        last_step = data[-1][0]
        current_step = first_step
        
        missing_steps = []
        data_index = 1
        while current_step < last_step:
            current_step += step_size

            if data[data_index][0] != current_step:
                missing_steps.append(current_step)

            if data[data_index][0] < current_step:
                data_index += 1

    def predict(self, unix_time):
        self.model.fit()
        self.model.summary()


if __name__ == "__main__":
    pass
