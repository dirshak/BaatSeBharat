import sys
import os
# Ensure TradingAgents is in path
_src_dir = os.path.dirname(os.path.abspath(__file__))
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
import numpy as np
import os

class DataFetcher:
    """Handles acquisition of global stock indices and speech metadata."""
    
    INDEX_MAP = {
        "India (Nifty 50)": "^NSEI",
        "USA (S&P 500)": "^GSPC",
        "EU (Euro Stoxx 50)": "^STOXX50E",
        "China (SSE Composite)": "000001.SS"
    }
    
    LEADER_MAP = {
        "Nirmala Sitharaman (India)": "nirmala_sitharaman",
        "Donald Trump (USA)": "donald_trump",
        "Ursula von der Leyen (EU)": "ursula_vdl",
        "Xi Jinping (China)": "xi_jinping"
    }

    @staticmethod
    def fetch_stock_data(source_name, start_date, end_date):
        """Fetches adjusted close prices for the selected global index."""
        ticker = DataFetcher.INDEX_MAP.get(source_name)
        if not ticker:
            raise ValueError(f"Unknown source: {source_name}")
            
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return pd.DataFrame()
            
        return df["Adj Close"].to_frame(name="Price")

    @staticmethod
    def get_speech_folder(leader_name):
        """Returns the local path for a leader's speeches."""
        slug = DataFetcher.LEADER_MAP.get(leader_name)
        if not slug:
            return "mann_ki_baat_transcripts" # Fallback
        return f"speeches/{slug}"

    @staticmethod
    def list_available_speeches(leader_name):
        """Mock version of speech discovery for global expansion."""
        folder = DataFetcher.get_speech_folder(leader_name)
        if not os.path.exists(folder):
            # Create dummy folder for demonstration if it doesn't exist
            # os.makedirs(folder, exist_ok=True)
            return []
        
        return [f for f in os.listdir(folder) if f.endswith(".txt")]
