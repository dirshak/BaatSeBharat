import sqlite3
import os

def get_db_connection(db_path):
    """
    Returns a SQLite database connection, ensuring the parent directory exists.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    # Connect and return with a higher timeout to handle concurrent access
    conn = sqlite3.connect(db_path, timeout=30)
    
    # Optionally enable foreign keys if needed by the schema
    conn.execute("PRAGMA foreign_keys = ON;")
    
    return conn
