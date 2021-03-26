import sqlite3
from sqlite3 import Error


def create_table():
    conn = sqlite3.connect("new.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS ChapterTS (id integer PRIMARY KEY,chapter_text text NOT NULL,summary text)""")
    conn.close()


if __name__ == '__main__':
    create_connection()