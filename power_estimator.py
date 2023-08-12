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
        total_power_output = [0, [0, 0]]
        for country, data in self._historical_data_source:
            try:
                p = PowerRegression(data)
                mean, ci = p.predict(time.time())
                #print(p.get_time_series())
                print(f"{country} - {mean}TW")

                total_power_output[0] += mean
                total_power_output[1][0] += ci[0]
                total_power_output[1][1] += ci[1]
            except ValueError:
                print(f"{country} has no data available")

        print(f"Global Power Consumption: {total_power_output}")

    def to_dict(self) -> typing.Dict:
        return self._historical_data_source.data



if __name__ == "__main__":
    from data_source import EiaJsonDataSource
    ds = EiaJsonDataSource("data/WorldWideYearlyEnergyConsumption.json")
    pe = PowerEstimator(ds)
