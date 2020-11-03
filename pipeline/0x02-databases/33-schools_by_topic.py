#!/usr/bin/env python3
"""
School by topic
"""


def schools_by_topic(mongo_collection, topic):
    """ returns the list of school having a specific topic.

    Args:
        mongo_collection: the pymongo collection object.
        topic (string): the topic searched.
    """
    return mongo_collection.find({"topics": {"$in": [topic]}})
