import time
import typing

from data_source import DataSource
from interpolation import NewtonPolynomialInterpolator


class PowerEstimator:

    # TODO: need to add live hooks eventually (use add_live function or something)
    def __init__(self, historical_data_source: DataSource):
        """ Initializes a Power Estimator to predict a given country's current power output as accurately as possible.
         If live data is not available, we fallback onto historical_data_source for predictions"""
        self._historical_data_source = historical_data_source

        self.__precompute_historical()

    def __precompute_historical(self):
        # COMPUTE NEWTON INTERPOLATION: https://en.wikipedia.org/wiki/Newton_polynomial
        # O(n^2) complexity and space - this might take a while with many data points
        # Input -> Data points
        # Output -> Newton Polynomial
        for country, data in self._historical_data_source:
            #print(f"{country} - {data}")
            n = NewtonPolynomialInterpolator(data)
            print(f"{country} - {n.predict(time.time())}")


    def to_dict(self) -> typing.Dict:
        return self._historical_data_source.data



if __name__ == "__main__":
    from data_source import EiaJsonDataSource
    ds = EiaJsonDataSource("data/WorldWideYearlyEnergyConsumption.json")
    pe = PowerEstimator(ds)