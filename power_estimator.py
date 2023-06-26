import time
import typing

from data_source import DataSource
from interpolation import NewtonPolynomialInterpolator
from auto_regression import PowerRegression


class PowerEstimator:

    # TODO: need to add live hooks eventually (use add_live function or something)
    def __init__(self, historical_data_source: DataSource):
        """ Initializes a Power Estimator to predict a given country's current power output as accurately as possible.
         If live data is not available, we fallback onto historical_data_source for predictions"""
        self._historical_data_source = historical_data_source

        self.__precompute_historical()

    def __precompute_historical(self):
        # Create an ARIMA model for each dataset
        # https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
        # Input -> Data points
        # Output -> Object with predict()/forecast() functionality
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for country, data in self._historical_data_source:
            #n = NewtonPolynomialInterpolator(data)
            p = PowerRegression(data)
            print(f"{country} - {p.predict(time.time())}")

            #print(f"{country} - {data}")
            # n = NewtonPolynomialInterpolator(data)
            # ax.plot(data[''])

            # ax.plot(n.x_data, n.y_data)
            # x_steps = [x for x in range(int(n.x_data[0]),int(time.time()),10000)]
            # ax.plot(x_steps, [n.predict(x) for x in x_steps])
            # plt.show()
            # print(f"{country} - {n.predict(time.time())} - {n.coefficients} - {data}")
            #input("CONTINUE?")

    def to_dict(self) -> typing.Dict:
        return self._historical_data_source.data



if __name__ == "__main__":
    from data_source import EiaJsonDataSource
    ds = EiaJsonDataSource("data/WorldWideYearlyEnergyConsumption.json")
    pe = PowerEstimator(ds)
