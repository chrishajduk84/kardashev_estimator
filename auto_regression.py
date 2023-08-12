import bisect
import copy
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
    b = y1
    # print(f"m: {m}, x: {x_interpolate}, b: {b}")
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
        self.interpolated_data = [interpolated_x, interpolated_y, copy.deepcopy(interpolated_y), copy.deepcopy(interpolated_y)]       # (x, y, y_low, y_high)
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

    def reset(self):
        self.__init__(self.data)

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

        predictions = []
        actual = []

        self.model = ARIMA(self.interpolated_data[1], order=(len(self.interpolated_data[1]), 0, len(self.interpolated_data[1])-1))
        self.fit = self.model.fit()

        # Model Print Debug
        # print(fit.summary())
        # residuals = DataFrame(fit.resid)
        # residuals.plot()
        # plt.show()
        # residuals.plot(kind='kde')
        # plt.show()
        # print(residuals.describe())

    def get_time_series(self, start_time=0, end_time=time.time()):
        """
        Process:
         * Figure out how many forecasted time steps need to be made
         * Interpolate the exact prediction between two time-steps
         * Return the exact prediction with a confidence interval
        :param unix_time: unix timestamp where the desired prediction should be made
        :return: tuple consisting of ( (x, y_predicted), (x, y_ci_low, y_ci_high) )
        """

        # Check if start and end_time is within existing data
        if start_time < 0 or end_time < 0:
            raise ValueError(f"Start time ({start_time}) or end time ({end_time}) before unix epoch is not supported")

        # Verify end time is after start time
        if start_time > end_time:
            raise ValueError(f"End time ({end_time}) precedes start time ({start_time})")

        if end_time - start_time < self.data_step:
            raise ValueError(f"Requested data range ({start_time} -> {end_time}) "
                             f"is smaller than the minimum step size {self.data_step}")

        # If the requested data is outside of the data stored in memory, lets try to predict it
        if end_time > self.interpolated_data[0][-1]:
            self.predict(end_time)

        # Create new tuple object from sub-array of self.interpolated_data
        start_index = bisect.bisect_left(self.interpolated_data[0], start_time)
        end_index = bisect.bisect_right(self.interpolated_data[0], end_time)

        data = [self.interpolated_data[0][start_index:end_index], self.interpolated_data[1][start_index:end_index]]

        limited_ci = [[], [], []]
        for i in range(len(self.interpolated_data[0])):
            if self.interpolated_data[1][i] != self.interpolated_data[2][i] or \
                    self.interpolated_data[1][i] != self.interpolated_data[3][i]:
                limited_ci[0].append(self.interpolated_data[0][i])
                limited_ci[1].append(self.interpolated_data[2][i])
                limited_ci[2].append(self.interpolated_data[3][i])

        return (data, limited_ci)

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
        # print(f"Predicting {number_of_predictions} time steps to achieve {unix_time} prediction")

        # Check if data already exists within self.interpolated_data, if so - we can skip the modeling and prediction step
        if number_of_predictions > 0:

            if self.model is None or self.fit is None:
                self.fit_model()

            # Model Forecasting:
            # OLD: yhat = fit.forecast(steps=number_of_predictions)
            # NEW: fit.get_forecast() returns PredictionResults object which contains .conf_int() and .predicted_mean (equivalent to y_hat)
            forecast = self.fit.get_forecast(steps=number_of_predictions)
            yhat = forecast.predicted_mean
            yhat_conf_interval = forecast.conf_int()

            if len(yhat) != len(yhat_conf_interval):
                raise AssertionError(f"yhat ({len(yhat)}) and yhat_conf_interval ({len(yhat_conf_interval)})"
                                     f" should be the same length")

            for i in range(len(yhat)):
                # Extend x-axis
                self.interpolated_data[0].append(self.interpolated_data[0][-1] + self.data_step)
                # Extend y-axis
                self.interpolated_data[1].append(yhat[i])
                # Extend low confidence interval
                self.interpolated_data[2].append(yhat_conf_interval[i][0])
                # Extend high confidence interval
                self.interpolated_data[3].append(yhat_conf_interval[i][1])

        # Figure out the data index to interpolate against
        for inter_index in range(len(self.interpolated_data[0])):
            if unix_time <= self.interpolated_data[0][inter_index]:
                break

        if inter_index == 0:
            raise AssertionError(f"inter_index should not be 0: {self.interpolated_data}")

        # Verify all data is above 0 (can't produce negative power)
        for i in range(len(self.interpolated_data[0])):
            if self.interpolated_data[1][i] < 0:
                self.interpolated_data[1][i] = 0
            if self.interpolated_data[2][i] < 0:
                self.interpolated_data[2][i] = 0
            if self.interpolated_data[3][i] < 0:
                self.interpolated_data[3][i] = 0

        # Interpolate between the closest two predictions/data points to get the predicted value for the requested time stamp
        x = unix_time - self.interpolated_data[0][inter_index-1]

        prediction = linear_interpolation(0, self.data_step, self.interpolated_data[1][inter_index-1], self.interpolated_data[1][inter_index], x)
        confidence_interval = [linear_interpolation(0, self.data_step, self.interpolated_data[2][inter_index-1], self.interpolated_data[2][inter_index], x),
                               linear_interpolation(0, self.data_step, self.interpolated_data[3][inter_index-1], self.interpolated_data[3][inter_index], x)]

        # Print debug
        # print(yhat)
        # print(yhat_conf_interval)
        # print(f"unix: {unix_time}, data_steps: {(self.interpolated_data[0][-1] + self.data_step * (number_of_predictions - 1))}")
        # print(prediction)
        # print(confidence_interval)
        #print(yhat[-1])
        #print(yhat[-2])

        # Plotting debug
        # fig, ax = plt.subplots()
        # plt.figure(1)
        # ax.plot(self.interpolated_data[0], self.interpolated_data[2], 'b-')
        # ax.plot(self.interpolated_data[0], self.interpolated_data[3], 'b-')
        # ax.plot(self.interpolated_data[0], self.interpolated_data[1], 'r+')
        # # ax.plot([self.interpolated_data[3][-i] for i in range(1, number_of_predictions+1)], yhat, 'bo')
        # ax.plot(unix_time, prediction, 'gx')
        # #plt.show()
        # plt.pause(0.1)

        if prediction < 0:
            prediction = 0

        for i in range(len(confidence_interval)):
            if confidence_interval[i] < 0:
                confidence_interval[i] = 0

        return prediction, confidence_interval


if __name__ == "__main__":
    pass
