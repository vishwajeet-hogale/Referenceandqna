import sqlite3
from sqlite3 import Error


def create_table():
    conn = sqlite3.connect("new.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS ChapterTS (id integer PRIMARY KEY AUTOINCREMENT,chapter_text text NOT NULL,topic text NOT NULL,summary text)""")
    conn.close()

def add_data(text,topic,summary):
    conn = sqlite3.connect("new.db")
    conn.execute("""INSERT INTO ChapterTS(chapter_text,topic,summary)
    VALUES("{}","{}","{}")""".format(text,topic,summary))
    conn.close()

if __name__ == '__main__':
    create_connection()