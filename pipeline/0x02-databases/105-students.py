#!/usr/bin/env python3
"""Sorting students"""


def top_students(mongo_collection):
    """returns all students sorted by average score.

    Args:
        mongo_collection: the pymongo collection object.
    """
    pipeline = [
        {"$unwind": "$topics"},
        {"$group": {"_id": "$_id", 'averageScore': {"$avg": "$topics.score"}}},
        {"$sort": {'averageScore': -1}}
    ]

    pipeline2 = [
        {"$project": {"_id": 1, "name": 1}}
    ]

    averages = mongo_collection.aggregate(pipeline)
    names = mongo_collection.aggregate(pipeline2)

    averages_list = [avg for avg in averages]
    names_list = [name for name in names]

    for avg in averages_list:
        for name in names_list:
            if avg['_id'] == name['_id']:
                avg['name'] = name['name']

    return averages_list
