#!/usr/bin/env python3
"""
Retrieving user location
"""
import requests
import sys
import time

if __name__ == '__main__':
    url = sys.argv[1]

    headers = {'Accept': 'application/vnd.github.v3+json'}
    r = requests.get(url, headers=headers)

    if r.status_code == 200:

        print(r.json()['location'])

    elif r.status_code == 404:
        print("Not found")

    elif r.status_code == 403:
        limit_time = int(r.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        remaining = int((limit_time - now) / 60)
        print("Reset in {} min".format(remaining))
