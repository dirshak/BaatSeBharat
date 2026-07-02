import asyncio
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
import pandas as pd
import yaml
import sqlite3
from datetime import datetime
import time
import sys
import os


def _ensure_event_loop():
    """Recreate asyncio event loop if Streamlit has closed it."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger
from src.utils.db_utils import get_db_connection

logger = setup_logger(__name__)

class MarketDataDownloader:
    """Download market data from Yahoo Finance"""
    
    def __init__(self, config_path='./config/market_tickers.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.sectors = config['sectors']
    
    def download_ticker_data(self, ticker_info, sector_name, start_date='2014-10-03'):
        """Download data for a single ticker"""
        symbol = ticker_info['symbol']
        
        logger.info(f"Downloading {symbol} ({sector_name})...")
        
        try:
            _ensure_event_loop()
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=datetime.now().strftime('%Y-%m-%d')
            )
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None
            
            # Reset index to get date as column
            df.reset_index(inplace=True)
            
            # Add metadata
            df['ticker'] = symbol
            df['sector'] = sector_name
            
            # Calculate returns
            df['returns'] = df['Close'].pct_change()
            
            # Calculate rolling volatility (20-day)
            df['volatility'] = df['returns'].rolling(window=20).std() * (252 ** 0.5)
            
            logger.info(f"Downloaded {len(df)} records for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return None
    
    def download_all_data(self):
        """Download data for all configured tickers"""
        all_data = []
        
        for sector in self.sectors:
            sector_name = sector['name']
            
            for ticker_info in sector['tickers']:
                df = self.download_ticker_data(ticker_info, sector_name)
                
                if df is not None:
                    all_data.append(df)
                
                # Rate limiting
                time.sleep(1)
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total records downloaded: {len(combined_df)}")
            return combined_df
        else:
            logger.error("No data downloaded")
            return None
    
    def save_to_database(self, df, db_path='./data/market_rhetoric.db'):
        """Save market data to database"""
        logger.info("Saving market data to database...")
        
        conn = get_db_connection(db_path)
        
        saved_count = 0
        
        for _, row in df.iterrows():
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO market_data
                    (date, ticker, sector, open, high, low, close, volume, returns, volatility)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['Date'].strftime('%Y-%m-%d'),
                    row['ticker'],
                    row['sector'],
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']),
                    float(row['returns']) if pd.notna(row['returns']) else None,
                    float(row['volatility']) if pd.notna(row['volatility']) else None
                ))
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving row: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {saved_count} market data records")
        return saved_count

if __name__ == "__main__":
    downloader = MarketDataDownloader()
    df = downloader.download_all_data()
    if df is not None:
        downloader.save_to_database(df)
