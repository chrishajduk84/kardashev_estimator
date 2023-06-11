import abc
import json
from country_code_converter import CountryCodeConverter

QBTU_TO_TWH = lambda x: x*293.07
HOURS_IN_YEAR = 8760

class DataSource(abc.ABC):
    """ Timestamps must be in unix seconds """
    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def data(self):
        pass

    @abc.abstractmethod
    def country_data(self, iso3_code):
        pass


class EiaJsonDataSource(DataSource):
    #https://www.eia.gov/international/data/world/total-energy/more-total-energy-data?pd=44&p=004000000000000000000000000000000000000000000000000000000f00o&u=2&f=A&v=mapbubble&a=-&i=none&vo=value&t=C&g=00000000000000000000000000000000000000000000000001&l=249-ruvvvvvfvtvnvv1vrvvvvfvvvvvvfvvvou20evvvvvvvvvvnvvvs0008&s=315532800000&e=1609459200000&ev=false&
    _data = {}
    _country_data = {}
    def __init__(self, file):
        # TODO future? Do we ever want multiple files?
        self._file = file
        self.parse()

    def __iter__(self):
        cc = CountryCodeConverter()
        for country in cc:
            yield country, self.country_data(country)

    def parse(self):
        with open(self._file, 'r') as f:
            raw_data = f.readlines()
        self._data = json.loads("".join(raw_data))

        for country_entry in self._data:
            if "iso" in country_entry and "data" in country_entry and "productid" in country_entry:
                if country_entry["productid"] != 44:
                    # For now we are only extracting total energy values
                    continue

                if country_entry["unit"] != "QBTU":
                    print(f"{country_entry} does not use QBTU units")

                data = []
                for data_entry in country_entry["data"]:
                    # EIA file has the date in milliseconds since unix epoch
                    # EIA file provides data in QBTU, which we convert to TWH -> TeraWatts

                    if data_entry["value"] == "(s)" or data_entry['value'] == "--" or data_entry['value'] == "-":
                        # Since the value is too small to register (s), or is not available ("--")
                        # or not applicable ("-") we will assume 0.
                        data_entry["value"] = 0


                    data.append((data_entry["date"]/1000, QBTU_TO_TWH(data_entry["value"])/HOURS_IN_YEAR))
                self._country_data[country_entry["iso"]] = data


    # def __next__(self):
    #     for country, data in self._country_data.items():
    #         yield country, data

    @property
    def data(self):
        return self._data

    def country_data(self, iso3_code):
        """ Get country data based on the iso3 country code provided """
        if iso3_code in self._country_data:
            return self._country_data[iso3_code]
        else:
            return []

