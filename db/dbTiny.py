from tinydb import TinyDB
from tinydb import Query, where


def init_db(path):
    db = TinyDB(path, indent=4, separators=(',', ': '))
    return db


def get_last(db, field):
    if len(db) == 0:
        return None
    post = sorted(db, key=lambda k: k['timestamp'])
    return post[-1][field]


def sort_by_field(db, field, reverse=False):
    if len(db) == 0:
        return None
    ordered = sorted(db, key=lambda k: k[field], reverse=reverse)
    return ordered


def insert(db, post):
    db.insert(post)
