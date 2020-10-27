#!/usr/bin/env python3
"""
Retrieving sentient planets
"""
import requests


def sentientPlanets():
    """
    returns the list of names of the home planets of all sentient species.
    """
    planets = []
    url = 'https://swapi-api.hbtn.io/api/species/'

    while url:
        r = requests.get(url)

        for specie in r.json()['results']:
            if specie['classification'] == 'sentient' or \
                    specie['designation'] == 'sentient':

                if specie['homeworld'] is not None:
                    homeworld = requests.get(specie['homeworld'])
                    planets.append(homeworld.json()['name'])

        url = r.json()['next']

    return planets
