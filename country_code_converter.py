import json

class CountryCodeConverter:

    def __init__(self):
        with open("data/country_codes.json", 'r') as f:
            self.country_data = json.load(f)

        # Generate quick-access mappings for converting to and from iso3 country codes
        self._iso3_to_country = {}
        for country in self.country_data:
            self._iso3_to_country[country["ISO3"]] = country["Country Name"]

        self._country_to_iso3 = {v: k for k, v in self._iso3_to_country.items()}

    def __iter__(self):
        for country in self._iso3_to_country.keys():
            yield country

    def to_iso3(self, name: str):
        return self._country_to_iso3.get(name.lower())

    def to_name(self, iso3):
        return self._iso3_to_country.get(iso3.upper())
