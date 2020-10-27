#!/usr/bin/env python3
"""
Retrieving user location
"""
import requests


if __name__ == '__main__':

    rockets = {}

    url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(url).json()

    for launch in launches:
        rocket_id = launch['rocket']
        rocket_url = "https://api.spacexdata.com/v4/rockets/{}".\
            format(rocket_id)
        rocket_name = requests.get(rocket_url).json()['name']

        if rocket_name in rockets.keys():
            rockets[rocket_name] += 1
        else:
            rockets[rocket_name] = 1

    sort_rockets = sorted(rockets.items(), key=lambda x: x[0], reverse=False)
    sort_rockets = sorted(sort_rockets, key=lambda x: x[1], reverse=True)

    for i in sort_rockets:
        print("{}: {}".format(i[0], i[1]))
