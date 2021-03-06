#!/usr/bin/env python3
"""
Retrieving user location
"""
import requests

if __name__ == '__main__':

    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    launches = requests.get(url).json()

    # Finding the next launch
    dates = [launch['date_unix'] for launch in launches]
    index = dates.index(min(dates))

    launch = launches[index]
    launch_name = launch['name']

    date = launch['date_local']

    rocket_id = launch['rocket']
    rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    rocket_name = requests.get(rocket_url).json()['name']

    launchpad_id = launch['launchpad']
    launchpad_url = "https://api.spacexdata.com/v4/launchpads/{}".\
        format(launchpad_id)
    launchpad_req = requests.get(launchpad_url).json()
    launchpad_name = launchpad_req['name']
    launchpad_loc = launchpad_req['locality']

    upcoming = "{} ({}) {} - {} ({})".format(launch_name,
                                             date,
                                             rocket_name,
                                             launchpad_name,
                                             launchpad_loc)

    print(upcoming)
