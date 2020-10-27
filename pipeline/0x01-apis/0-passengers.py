#!/usr/bin/env python3
"""
Retrieving passengers
"""
import requests


def availableShips(passengerCount):
    """returns the list of ships that can hold a given number of passengers.

    Args:
        passengerCount (int): number of passengers to hold

    Returns:
        list of ships.
    """
    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships/'

    while url:
        r = requests.get(url)

        for ship in r.json()['results']:
            passengers = ship['passengers'].replace(',', '')

            if passengers.isnumeric() and int(passengers) >= passengerCount:
                ships.append(ship['name'])

        url = r.json()['next']

    return ships
