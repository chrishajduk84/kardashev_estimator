from flask import Flask, request, jsonify, abort
from power_estimator import PowerEstimator
from data_source import EiaJsonDataSource
import json

################################################
# kardeshev_estimator
################################################
# Conversion from power (in Watts) to the Kardashev scale will occur in the browser via a simple javascript converstion
# This app is responsible for:
# Collecting and aggregating the latest possible power data and then serving it via the REST endpoints

app = Flask(__name__)
ds = EiaJsonDataSource("data/WorldWideYearlyEnergyConsumption.json")
pe = PowerEstimator(historical_data_source=ds)


class CountryCodeConvertercountry:
    pass


@app.get("/power/")
@app.get("/power/<country>")
def get_power(country=None):
    """ Will calculate and estimate total energy consumption of the entire world
        returns json object containing total power and breakdown by country
    """
    # TODO: Update cached data with live data

    if country is None:
        return json.dumps(pe.to_dict())

    if len(country) == 3:
        return json.dumps(pe[country])
    else:
        code = CountryCodeConvertercountry().to_iso3(country)
        try:
            return json.dumps(pe[code])
        except ValueError as e:
            abort(404)
    # # Get cached data from the PowerEstimator
    # print(ds.country_data("CAN"))
    # return json.dumps(pe.to_dict())



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)