#!/usr/bin/env python3
"""
nginx logs
"""
from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient()
    collection = client.logs.nginx

    # cursor = collection.find({})
    # print(cursor[0])

    number_logs = collection.count_documents({})
    print("{} logs".format(number_logs))

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        query = {"method": method}
        count_methods = collection.count_documents(query)
        print("\tmethod {}: {}".format(method, count_methods))

    query2 = {"method": "GET", "path": "/status"}
    count2 = collection.count_documents(query2)
    print("{} status check".format(count2))
