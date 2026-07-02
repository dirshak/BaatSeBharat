import sys
import os
# Ensure TradingAgents is in path
_data_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_data_dir)
_root_dir = os.path.dirname(_src_dir)
_ta_dir = os.path.join(_root_dir, 'TradingAgents')
if _ta_dir not in sys.path:
    sys.path.insert(0, _ta_dir)
try:
    from tradingagents.dataflows import yf_cache_patch
except Exception:
    pass
import yfinance as yf
import sqlite3
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class VIXDownloader:
    """Download VIX (Volatility Index) data"""
    
    def __init__(self):
        self.symbol = '^VIX'
    
    def download_vix(self, start_date='2014-10-03'):
        """Download VIX data"""
        logger.info(f"Downloading VIX data from {start_date}...")
        
        try:
            vix = yf.Ticker(self.symbol)
            df = vix.history(
                start=start_date,
                end=datetime.now().strftime('%Y-%m-%d')
            )
            
            if df.empty:
                logger.error("No VIX data downloaded")
                return None
            
            df.reset_index(inplace=True)
            logger.info(f"Downloaded {len(df)} VIX records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading VIX: {e}")
            return None
    
    def save_to_database(self, df, db_path='./data/market_rhetoric.db'):
        """Save VIX data to database"""
        logger.info("Saving VIX data to database...")
        
        conn = sqlite3.connect(db_path)
        
        saved_count = 0
        
        for _, row in df.iterrows():
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO vix_data
                    (date, vix_open, vix_high, vix_low, vix_close)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    row['Date'].strftime('%Y-%m-%d'),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close'])
                ))
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving VIX row: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {saved_count} VIX records")
        return saved_count

if __name__ == "__main__":
    downloader = VIXDownloader()
    df = downloader.download_vix()
    if df is not None:
        downloader.save_to_database(df)
