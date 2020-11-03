#!/usr/bin/env python3
"""
List Documents
"""


def list_all(mongo_collection):
    """lists all documents in a collection.

    Args:
        mongo_collection:  the pymongo collection object.

    Returns:
        list with documents.
    """
    return mongo_collection.find({})
