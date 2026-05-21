import sqlite3
import pandas as pd
from datetime import datetime, timedelta
 
# Constants
DB_PATH = "chatbot_data.db"
FRESHNESS_CURRENT_PRICE = 60        # 1 minute
FRESHNESS_DAILY_HISTORY = 82800     # 23 hours -- daily OHLCV only updates once after market close
 
 
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
 
def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS current_prices (
            ticker TEXT PRIMARY KEY,
            price REAL,
            currency TEXT,
            company_name TEXT,
            timestamp DATETIME
        )
    ''')
 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_history (
            ticker TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    ''')
 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fetch_log (
            ticker TEXT,
            data_type TEXT,
            timestamp DATETIME,
            PRIMARY KEY (ticker, data_type)
        )
    ''')
 
    conn.commit()
    conn.close()
 
 
def save_price_history(ticker, df, table_name="daily_history"):
    conn = get_db_connection()
    cursor = conn.cursor()
 
    for _, row in df.iterrows():
        cursor.execute(f'''
            INSERT OR REPLACE INTO {table_name} (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            row.name.date(),
            row['Open'],
            row['High'],
            row['Low'],
            row['Close'],
            row['Volume']
        ))
 
    cursor.execute('''
        INSERT OR REPLACE INTO fetch_log (ticker, data_type, timestamp)
        VALUES (?, ?, ?)
    ''', (ticker, table_name, datetime.now()))
 
    conn.commit()
    conn.close()
 
 
def load_price_history(ticker, table_name="daily_history"):
    conn = get_db_connection()
    cursor = conn.cursor()
 
    cursor.execute(f'''
        SELECT date, open, high, low, close, volume
        FROM {table_name}
        WHERE ticker = ?
        ORDER BY date
    ''', (ticker,))
 
    rows = cursor.fetchall()
    conn.close()
 
    if not rows:
        return pd.DataFrame()
 
    df = pd.DataFrame(rows, columns=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df
 
 
def is_data_fresh(ticker, data_type, freshness_threshold):
    conn = get_db_connection()
    cursor = conn.cursor()
 
    cursor.execute('''
        SELECT timestamp
        FROM fetch_log
        WHERE ticker = ? AND data_type = ?
    ''', (ticker, data_type))
 
    row = cursor.fetchone()
    conn.close()
 
    if row is None:
        return False
 
    try:
        last_fetch = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        last_fetch = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
 
    return datetime.now() - last_fetch < timedelta(seconds=freshness_threshold)