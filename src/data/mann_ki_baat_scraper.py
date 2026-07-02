import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime
import sqlite3
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MannKiBaatScraper:
    """Scraper for Mann Ki Baat transcripts"""
    
    def __init__(self, base_url="https://www.pmindia.gov.in/en/mann-ki-baat/"):
        self.base_url = base_url
        self.episodes = []
        
    def fetch_episode_list(self):
        """Fetch list of all Mann Ki Baat episodes"""
        logger.info("Fetching episode list...")
        
        try:
            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all episode links
            episode_links = soup.find_all('a', href=True)
            
            episodes = []
            for link in episode_links:
                if 'mann-ki-baat' in link['href']:
                    episodes.append({
                        'url': link['href'],
                        'title': link.text.strip()
                    })
            
            logger.info(f"Found {len(episodes)} episodes")
            return episodes
            
        except Exception as e:
            logger.error(f"Error fetching episode list: {e}")
            return []
    
    def fetch_episode_content(self, episode_url):
        """Fetch content of a single episode"""
        try:
            response = requests.get(episode_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract date
            date_element = soup.find('time') or soup.find('span', class_='date')
            date_str = date_element.text.strip() if date_element else None
            
            # Extract full text
            content_div = soup.find('div', class_='content') or soup.find('article')
            full_text = content_div.get_text(separator='\n').strip() if content_div else ""
            
            return {
                'date': self.parse_date(date_str),
                'full_text': full_text,
                'url': episode_url
            }
            
        except Exception as e:
            logger.error(f"Error fetching episode {episode_url}: {e}")
            return None
    
    def parse_date(self, date_str):
        """Parse date from various formats"""
        if not date_str:
            return None
        
        # Try different date formats
        formats = [
            '%B %d, %Y',
            '%d %B %Y',
            '%Y-%m-%d',
            '%d-%m-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def scrape_all_episodes(self, delay=2):
        """Scrape all episodes with rate limiting"""
        episodes_list = self.fetch_episode_list()
        
        all_episodes = []
        
        for i, episode_info in enumerate(episodes_list, 1):
            logger.info(f"Scraping episode {i}/{len(episodes_list)}: {episode_info['title']}")
            
            content = self.fetch_episode_content(episode_info['url'])
            
            if content:
                content['title'] = episode_info['title']
                all_episodes.append(content)
            
            # Rate limiting
            time.sleep(delay)
        
        self.episodes = all_episodes
        logger.info(f"Successfully scraped {len(all_episodes)} episodes")
        
        return all_episodes
    
    def save_to_database(self, db_path='./data/market_rhetoric.db'):
        """Save scraped episodes to database"""
        logger.info("Saving episodes to database...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        
        for i, episode in enumerate(self.episodes, 1):
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO speeches 
                    (date, source, country, title, full_text, language, episode_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    episode.get('date'),
                    'Mann Ki Baat',
                    'India',
                    episode.get('title'),
                    episode.get('full_text'),
                    'Hindi/English',
                    i
                ))
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving episode {i}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {saved_count} episodes to database")
        return saved_count

if __name__ == "__main__":
    scraper = MannKiBaatScraper()
    episodes = scraper.scrape_all_episodes()
    scraper.save_to_database()
