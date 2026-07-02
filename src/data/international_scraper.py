import requests
from bs4 import BeautifulSoup
import yaml
import sqlite3
from datetime import datetime
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class InternationalSpeechScraper:
    """Scraper for international leadership speeches"""
    
    def __init__(self, config_path='./config/international_sources.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.sources = config['sources']
        self.speeches = []
    
    def scrape_source(self, source):
        """Scrape speeches from a single source"""
        logger.info(f"Scraping {source['country']}...")
        
        # Implementation depends on specific website structure
        # This is a template
        
        try:
            response = requests.get(source['url'], timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find speech links (site-specific)
            speech_links = soup.find_all('a', class_='speech-link')
            
            speeches = []
            for link in speech_links[:10]:  # Limit to recent 10
                speech_data = self.fetch_speech_content(link['href'])
                if speech_data:
                    speech_data['country'] = source['country']
                    speech_data['weight'] = source['weight']
                    speeches.append(speech_data)
                
                time.sleep(2)  # Rate limiting
            
            return speeches
            
        except Exception as e:
            logger.error(f"Error scraping {source['country']}: {e}")
            return []
    
    def fetch_speech_content(self, url):
        """Fetch individual speech content"""
        # Similar to Mann Ki Baat scraper
        pass
    
    def scrape_all_sources(self):
        """Scrape all configured sources"""
        for source in self.sources:
            source_speeches = self.scrape_source(source)
            self.speeches.extend(source_speeches)
        
        logger.info(f"Total speeches collected: {len(self.speeches)}")
        return self.speeches
    
    def save_to_database(self, db_path='./data/market_rhetoric.db'):
        """Save to database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for speech in self.speeches:
            cursor.execute('''
                INSERT OR REPLACE INTO speeches
                (date, source, country, title, full_text, language)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                speech.get('date'),
                'International',
                speech['country'],
                speech.get('title'),
                speech.get('full_text'),
                'English'
            ))
        
        conn.commit()
        conn.close()
