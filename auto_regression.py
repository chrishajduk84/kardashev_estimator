import math
import time

from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

def differenced_series(data, interval=1):
    diff = []
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return diff

def linear_interpolation(x1, x2, y1, y2, x_interpolate):
    # Interpolate to get the predicted value for the requested x_interpolate
    m = (y2 - y1) / (x2 - x1)
    # x = unix_time - (self.interpolated_data[0][-1] + self.data_step * (number_of_predictions - 1))
    b = y1
    print(f"m: {m}, x: {x_interpolate}, b: {b}")
    # print(f"unix: {unix_time}, data_steps: {(self.interpolated_data[0][-1] + self.data_step * (number_of_predictions - 1))}")
    prediction = m * x_interpolate + b
    return prediction


class PowerRegression:
    
    def __init__(self, data):
        """ The initialization of this class must contain a list of (x,y) tuple data. This class will ensure the data is linearly interpolated to create a constant periodicity. """

        if len(data) == 0 or len(data[0]) == 0:
            raise ValueError("Insufficient data provided for PowerRegression")

        self.data = data
        self.data_step = self.__min_data_step(self.data)
        # Are any data steps missing? Do we need to interpolate for those?
        interpolated_y, interpolated_x = self.__interpolate_data_steps(self.data, self.data_step)
        print(self.data_step)
        #print(self.data)
        print(interpolated_y)
        print(interpolated_x)
        self.interpolated_data = (interpolated_x, interpolated_y)
        self.model = None
        self.fit = None


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
        
        new_steps = [first_step]
        new_values = [data[0][1]]
        data_index = 1
        while current_step + step_size < last_step:
            current_step += step_size

            # if data[data_index][0] != current_step:
            #     # We have found a missing step

            if data[data_index][0] < current_step:
                data_index += 1

            # y = mx + b interpolation
            m = (data[data_index][1] - data[data_index-1][1])/(data[data_index][0] - data[data_index-1][0])
            x = (current_step-data[data_index-1][0])
            b = data[data_index-1][1]
            intervalue = m*x + b
            new_steps.append(current_step)
            new_values.append(intervalue)
        return new_values, new_steps

    def fit_model(self):
        """
        ARIMA model uses 3 parameters (in addition to the data)
        p = the number of data points to complete the regression on
        d = integration order (d=1 means f(t) - f(t-1), d=2 would use double that difference)
        q = the averaging window
        order = (p,d,q)

        x_data = self.interpolated_data[0]
        y_data = self.interpolated_data[1]

        :return: None
        """

        fig, ax = plt.subplots()

        ax.plot(self.interpolated_data[0], self.interpolated_data[1])
        #x_steps = [x for x in range(int(interpolated_x),int(time.time()),10000)]
        #ax.plot(x_steps, [n.predict(x) for x in x_steps])
        plt.show()
        predictions = []
        actual = []

        self.model = ARIMA(self.interpolated_data[1], order=(len(self.interpolated_data[1]), 0, len(self.interpolated_data[1])-1))
        self.fit = self.model.fit()

    def predict(self, unix_time):
        """
        Process:
         * Figure out how many forecasted time steps need to be made
         * Interpolate the exact prediction between two time-steps
         * Return the exact prediction with a confidence interval
        :param unix_time: unix timestamp where the desired prediction should be made
        :return: tuple consisting of (predicted value, [confidence interval])
        """

        number_of_predictions = math.ceil((unix_time - self.interpolated_data[0][-1])/self.data_step)
        print(f"Predicting {number_of_predictions} time steps to achieve {unix_time} prediction")

        if self.model is None or self.fit is None:
            self.fit_model()

        # Model Forecasting:
        # OLD: yhat = fit.forecast(steps=number_of_predictions)
        # NEW: fit.get_forecast() returns PredictionResults object which contains .conf_int() and .predicted_mean (equivalent to y_hat)
        forecast = self.fit.get_forecast(steps=number_of_predictions)
        yhat = forecast.predicted_mean
        yhat_conf_interval = forecast.conf_int()

        print(yhat)
        print(yhat_conf_interval)
        #print(fit.summary())
        #residuals = DataFrame(fit.resid)
        #residuals.plot()
        # plt.show()
        #residuals.plot(kind='kde')
        # plt.show()
        #print(residuals.describe())

        # Interpolate to get the predicted value for the requested time stamp
        # m = (yhat[-1] - yhat[-2])/self.data_step
        #
        x = unix_time - (self.interpolated_data[0][-1] + self.data_step*(number_of_predictions - 1))
        # b = yhat[-2]
        # print(f"m: {m}, x: {x}, b: {b}")
        prediction = linear_interpolation(0, self.data_step, yhat[-2], yhat[-1], x)
        confidence_interval = [linear_interpolation(0, self.data_step, yhat_conf_interval[-2][0], yhat_conf_interval[-1][0], x),
                               linear_interpolation(0, self.data_step, yhat_conf_interval[-2][1], yhat_conf_interval[-1][1], x)]

        print(f"unix: {unix_time}, data_steps: {(self.interpolated_data[0][-1] + self.data_step*(number_of_predictions - 1))}")
        # prediction = m*x + b
        print(prediction)
        print(confidence_interval)
        #print(yhat[-1])
        #print(yhat[-2])

        # Plotting debug
        plt.plot([self.interpolated_data[0][-1] + self.data_step*i for i in range(1, number_of_predictions+1)], yhat, 'bo')
        plt.plot(self.interpolated_data[0], self.interpolated_data[1], 'r+')
        plt.plot(unix_time, prediction, 'gx')
        plt.show()

        if prediction < 0:
            prediction = 0

        for i in range(len(confidence_interval)):
            if confidence_interval[i] < 0:
                confidence_interval[i] = 0

        return prediction, confidence_interval


if __name__ == "__main__":
    pass
