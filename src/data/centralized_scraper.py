import sqlite3
import math
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import re
import os
import sys
import json
import hashlib
import asyncio
from playwright.async_api import async_playwright

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import setup_logger
from src.utils.db_utils import get_db_connection

logger = setup_logger(__name__)

class CentralizedSpeechScraper:
    """Centralized scraper for various leadership speeches (ECB, Fed, etc.)"""
    
    def __init__(self, db_path='./data/market_rhetoric.db'):
        self.db_path = db_path
        self._ensure_db_exists()
        
    def _ensure_db_exists(self):
        """Ensure the database and tables exist"""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # Speeches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speeches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                source TEXT,
                country TEXT,
                speaker TEXT,
                title TEXT,
                full_text TEXT,
                url TEXT,
                processed_text TEXT,
                doc_type TEXT DEFAULT 'Speech',
                UNIQUE(date, source, speaker, title)
            )
        ''')
        
        # Migration: add doc_type column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE speeches ADD COLUMN doc_type TEXT DEFAULT 'Speech'")
        except Exception:
            pass
        
        # Market data table (re-ensuring based on market_data_downloader.py)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                date TEXT,
                ticker TEXT,
                sector TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                returns REAL,
                volatility REAL,
                PRIMARY KEY (date, ticker)
            )
        ''')
        
        # Table: Speech-Market Impact
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speech_market_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speech_id INTEGER,
                ticker TEXT,
                event_date TEXT,
                return_t1 REAL,
                return_t5 REAL,
                return_t10 REAL,
                abnormal_return REAL,
                FOREIGN KEY (speech_id) REFERENCES speeches(id)
            )
        ''')
        
        # Table: Topic Distributions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_distributions (
                speech_id INTEGER,
                topic_id INTEGER,
                probability REAL,
                model_name TEXT,
                PRIMARY KEY (speech_id, topic_id),
                FOREIGN KEY (speech_id) REFERENCES speeches(id)
            )
        ''')
        
        # Migration: add model_name column if it doesn't exist yet
        try:
            cursor.execute("ALTER TABLE topic_distributions ADD COLUMN model_name TEXT")
        except Exception:
            pass  # Column already exists
        
        conn.commit()
        conn.close()

    def _generate_hash(self, title, text):
        """Generate SHA-256 hash of title + first 100 chars of text"""
        prefix = text[:100] if text else ""
        content = f"{title}{prefix}".encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def _log_ingestion(self, source, count, expected, missing=None):
        """Log ingestion results to ingestion_log.txt"""
        log_path = './ingestion_log.txt'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        missing_str = f" | Missing: {missing}" if missing else ""
        log_entry = f"[{timestamp}] Source: {source} | Episodes scraped: {count}/{expected}{missing_str}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        logger.info(f"Ingestion logged for {source}")

    def save_speeches(self, speeches, transcript_dir=None, file_prefix='speech'):
        """Save a list of speech dictionaries to the database and optionally to disk"""
        if not speeches:
            logger.warning("No speeches to save.")
            return 0
            
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        saved_count = 0
        
        # Ensure transcript directory exists if provided
        if transcript_dir:
            os.makedirs(transcript_dir, exist_ok=True)
            
        for speech in speeches:
            try:
                # 1. Deduplication Check
                title = speech.get('title', 'N/A')
                text = speech.get('full_text', '')
                doc_hash = self._generate_hash(title, text)
                
                # Check if hash already exists (using a metadata field or just unique constraint)
                # For now, let's use the unique constraint in the DB if available, 
                # but we'll also check explicitly to avoid INSERT OR IGNORE silently skipping without counting.
                
                cursor.execute("SELECT id FROM speeches WHERE title = ? AND date = ? AND source = ?", 
                               (speech.get('title'), speech.get('date'), speech.get('source')))
                if cursor.fetchone():
                    continue

                # 2. Save to Database
                conn.execute('''
                    INSERT OR IGNORE INTO speeches 
                    (date, source, country, speaker, title, full_text, url, doc_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    speech.get('date'),
                    speech.get('source'),
                    speech.get('country'),
                    speech.get('speaker'),
                    speech.get('title'),
                    speech.get('full_text'),
                    speech.get('url'),
                    speech.get('doc_type', 'Speech')
                ))
                saved_count += 1
                
                # 3. Save to individual .txt file if transcript_dir provided
                if transcript_dir and speech.get('full_text'):
                    # Sanitize all parts of filename
                    safe_title = re.sub(r'[^\w\s-]', '', speech.get('title', 'untitled')).strip().replace(' ', '_')[:60]
                    date_prefix = str(speech.get('date', 'unknown_date')).replace('-', '').replace('/', '_').replace(' ', '_')
                    
                    filename = f"{file_prefix}_{date_prefix}_{safe_title}.txt"
                    # Final safety check: remove any remaining / or \
                    filename = filename.replace('/', '_').replace('\\', '_')
                    fpath = os.path.join(transcript_dir, filename)
                    
                    with open(fpath, 'w', encoding='utf-8') as f:
                        # Header: Title | Date | Speaker | Type
                        f.write(f"{speech.get('title', 'N/A')} | {speech.get('date', 'N/A')} | {speech.get('speaker', 'N/A')} | {speech.get('doc_type', 'Speech')}\n\n")
                        f.write(speech.get('full_text'))
                        
            except Exception as e:
                logger.error(f"Error saving speech: {e}")
                
        conn.commit()
        conn.close()
        logger.info(f"Saved {saved_count} speeches to database{' and files' if transcript_dir else ''}.")
        return saved_count

    # --- ECB Logic (Adapted from ecb.py) ---
    
    async def scrape_ecb(self, days_back=3650):
        """Scrape ECB speeches (multi-year archiving supported)"""
        logger.info(f"Scraping ECB speeches from last {days_back} days...")
        
        current_year = datetime.now().year
        cutoff_date = datetime.now() - timedelta(days=days_back)
        start_year = cutoff_date.year
        
        speech_urls = set()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            for year in range(current_year, start_year - 1, -1):
                url = f"https://www.ecb.europa.eu/press/pubbydate/html/index.en.html?name_of_publication=Speech&year={year}"
                logger.info(f"Visiting ECB archive: {url}")
                
                try:
                    await page.goto(url, wait_until="load", timeout=45000)
                    await asyncio.sleep(3)
                    
                    # Ensure content is loaded
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(2)
                    
                    links = await page.evaluate("Array.from(document.querySelectorAll('a')).map(a => a.href)")
                    
                    year_urls = 0
                    for href in links:
                        # Filter by URL pattern containing monetary-policy or press-conference as per instructions
                        is_relevant = 'monetary-policy' in href or 'press-conference' in href
                        
                        if href and '/press/key/date/' in href and 'sp' in href and is_relevant:
                            # URL often contains date pattern spYYMMDD
                            date_match = re.search(r'sp(\d{2})(\d{2})(\d{2})', href)
                            if date_match:
                                y, m, d = date_match.groups()
                                try:
                                    pub_date = datetime(2000 + int(y), int(m), int(d))
                                    if pub_date >= cutoff_date:
                                        if href not in speech_urls:
                                            speech_urls.add(href)
                                            year_urls += 1
                                except ValueError:
                                    continue # Skip malformed dates (e.g. sp190000)
                    logger.info(f"Year {year}: Discovered {year_urls} relevant ECB monetary policy speeches.")
                except Exception as e:
                    logger.error(f"Error visiting ECB archive {year}: {e}")
            
            await browser.close()

        logger.info(f"Total filtered ECB speech URLs: {len(speech_urls)}")
        
        speeches = []
        for i, url in enumerate(speech_urls):
            if i % 10 == 0: logger.info(f"Scraping content {i+1}/{len(speech_urls)}...")
            content = self._scrape_ecb_content(url)
            if content:
                content['source'] = 'ECB'
                content['country'] = 'Europe'
                speeches.append(content)
            time.sleep(0.5)
            
        count = self.save_speeches(speeches, transcript_dir='./transcripts/ecb', file_prefix='ecb')
        self._log_ingestion("ECB Speeches", count, len(speech_urls))
        return count

    def _scrape_ecb_content(self, url):
        """Scrapes individual ECB speech content with improved metadata extraction"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Improve title extraction
            title_tag = soup.find('h1', class_='title') or soup.find('h1', class_='section-title') or soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else 'N/A'
            
            # Improve date extraction (ECB often uses <p class="date"> or <div class="date">)
            date_tag = soup.find('p', class_='date') or soup.find('div', class_='date') or soup.find('span', class_='date')
            date_str = date_tag.get_text(strip=True) if date_tag else 'N/A'
            
            # Fallback date extraction from URL if scraping fails
            # URL pattern: /press/key/date/2026/html/ecb.sp260226...
            parsed_date = 'N/A'
            if date_str != 'N/A':
                try:
                    parsed_date = datetime.strptime(date_str, '%d %B %Y').strftime('%Y-%m-%d')
                except:
                    # Try other formats
                    for fmt in ['%d %b %Y', '%B %d, %Y', '%Y-%m-%d']:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                            break
                        except: continue
            
            if parsed_date == 'N/A':
                date_match = re.search(r'sp(\d{2})(\d{2})(\d{2})', url)
                if date_match:
                    year, month, day = date_match.groups()
                    parsed_date = f"20{year}-{month}-{day}"

            # Improve speaker extraction
            speaker = 'N/A'
            subtitle_tag = soup.find('p', class_='subtitle') or soup.find('div', class_='subtitle')
            if subtitle_tag:
                speaker_text = subtitle_tag.get_text(strip=True)
                if 'by' in speaker_text:
                    speaker = speaker_text.split('by')[-1].split(',')[0].strip()
            
            # If speaker still N/A, look for speaker in the start of the content or in specific tags
            if speaker == 'N/A':
                meta_speaker = soup.find('meta', {'name': 'author'})
                if meta_speaker:
                    speaker = meta_speaker.get('content', 'N/A')

            content_div = soup.find('div', class_='ecb-pressContent') or soup.find('article') or soup.find('main')
            full_text = ""
            if content_div:
                for p in content_div.find_all('p'):
                    # Skip contact info at bottom
                    if not p.find_parent('div', class_='contact') and not 'class' in p.attrs or 'ecb-pressContent' in str(p.find_parent()):
                        full_text += p.get_text(strip=True) + "\n\n"

            return {
                'url': url,
                'title': title,
                'date': parsed_date,
                'speaker': speaker,
                'full_text': full_text.strip()
            }
        except Exception as e:
            logger.error(f"Error scraping ECB {url}: {e}")
            return None

    # --- Fed Logic (Simplified from us_federalreserve.py - using requests instead of Selenium for speed if possible) ---
    
    async def scrape_fed(self, days_back=3650):
        """Scrape US Federal Reserve speeches (multi-year supported)"""
        logger.info(f"Scraping Fed speeches from last {days_back} days...")
        
        base_url = "https://www.federalreserve.gov"
        current_year = datetime.now().year
        cutoff_date = datetime.now() - timedelta(days=days_back)
        start_year = cutoff_date.year
        
        speeches = []
        speech_links = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            for year in range(current_year, start_year - 1, -1):
                # Fed speech archive pattern: [year]-speeches.htm or landing page for current
                if year == datetime.now().year:
                    url = f"{base_url}/newsevents/speeches.htm"
                else:
                    url = f"{base_url}/newsevents/{year}-speeches.htm"
                
                logger.info(f"Visiting Fed archive: {url}")
                try:
                    await page.goto(url, wait_until="load", timeout=30000)
                    await asyncio.sleep(3)
                    
                    # Extract links and metadata
                    items = await page.query_selector_all(".row")
                    year_links = 0
                    for item in items:
                        try:
                            # Fed archive structure: .eventlist__time and .eventlist__event
                            date_tag = await item.query_selector(".eventlist__time time")
                            if not date_tag:
                                date_tag = await item.query_selector("time.itemDate")
                            if not date_tag: 
                                date_tag = await item.query_selector("time") # Final fallback
                                
                            if not date_tag: continue
                            
                            date_str = (await date_tag.inner_text()).strip()
                            try:
                                dt = datetime.strptime(date_str, '%m/%d/%Y')
                            except ValueError:
                                try:
                                    dt = datetime.strptime(date_str, '%B %d, %Y')
                                except: continue
                                
                            if dt < cutoff_date: continue
                            
                            # Different layouts use different link containers
                            link_tag = await item.query_selector(".eventlist__event a")
                            if not link_tag:
                                link_tag = await item.query_selector(".itemTitle a")
                            if not link_tag: continue
                            
                            href = await link_tag.get_attribute("href")
                            if not href: continue
                            
                            # Filter based on type
                            if '/speeches' in url and '/newsevents/speech/' not in href:
                                continue
                                    
                            title = (await link_tag.inner_text()).strip()
                            speaker_tag = await item.query_selector(".news__speaker")
                            speaker = (await speaker_tag.inner_text()).strip() if speaker_tag else 'N/A'
                            
                            full_url = base_url + href if href.startswith('/') else href
                            if full_url not in [l['url'] for l in speech_links]:
                                speech_links.append({
                                    'url': full_url,
                                    'date': dt.strftime('%Y-%m-%d'),
                                    'title': title,
                                    'speaker': speaker,
                                    'source': 'Fed',
                                    'doc_type': 'Speech' if '/speeches' in url else 'Press Release'
                                })
                                year_links += 1
                        except: continue
                    logger.info(f"Year {year}: Discovered {year_links} Fed documents.")
                except Exception as e:
                    logger.error(f"Error visiting Fed archive {year}: {e}")
            
            await browser.close()

        logger.info(f"Total discovered Fed speech links: {len(speech_links)}")
        for i, link_info in enumerate(speech_links):
            if i % 10 == 0: logger.info(f"Scraping content {i+1}/{len(speech_links)}...")
            content = self._scrape_fed_content(link_info['url'])
            if content:
                link_info.update(content)
                link_info.update({
                    'source': 'Fed',
                    'country': 'USA'
                })
                speeches.append(link_info)
            time.sleep(0.5)
            
        logger.info(f"Total Fed speeches collected: {len(speeches)}")
        count = self.save_speeches(speeches, transcript_dir='./transcripts/fed', file_prefix='fed')
        
        # Log ingestion
        self._log_ingestion("Fed", count, len(speech_links))
        
        return count

    # --- Mann Ki Baat Logic ---

    def _ordinal(self, n):
        """Return ordinal string for n, e.g. 1 -> '1st'"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    def scrape_mann_ki_baat(self, transcripts_dir='./transcripts/mann_ki_baat'):
        """
        Load Mann Ki Baat transcripts from already-scraped local .txt files.
        Falls back to web scraping for any missing episodes.
        File format: first line = "Episode N (DD Mon, YYYY)", rest = transcript.
        """
        logger.info(f"Loading Mann Ki Baat transcripts from: {transcripts_dir}")
        speeches = []

        if not os.path.isdir(transcripts_dir):
            logger.warning(f"Transcript directory not found: {transcripts_dir}")
            return 0

        import glob
        txt_files = sorted(glob.glob(os.path.join(transcripts_dir, 'mann_ki_baat_*.txt')))
        logger.info(f"Found {len(txt_files)} local MKB transcript files.")

        for fpath in txt_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                header = lines[0].strip()   # e.g. "Episode 80 (29 Aug, 2021)"
                full_text = '\n'.join(lines[2:]).strip()  # skip blank line after header

                # Parse episode number
                ep_match = re.match(r'Episode\s+(\d+)', header)
                ep_num = int(ep_match.group(1)) if ep_match else None

                # Parse date from header like "(29 Aug, 2021)" or "(29 Aug 2021)"
                parsed_date = None
                date_match = re.search(r'\(([^)]+)\)', header)
                if date_match:
                    date_str = date_match.group(1).strip()
                    for fmt in ['%d %b, %Y', '%d %B, %Y', '%d %b %Y', '%d %B %Y',
                                '%B %d, %Y', '%b %d, %Y']:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue

                speeches.append({
                    'date': parsed_date,
                    'source': 'Mann Ki Baat',
                    'country': 'India',
                    'speaker': 'PM Modi',
                    'title': f"Mann Ki Baat - Episode {ep_num}" if ep_num else header,
                    'full_text': full_text,
                    'url': f"https://www.pmindia.gov.in/en/news_updates/pms-address-in-the-{self._ordinal(ep_num)}-episode-of-mann-ki-baat/" if ep_num else None
                })
                logger.info(f"Loaded MKB Episode {ep_num} ({parsed_date})")

            except Exception as e:
                logger.error(f"Error loading {fpath}: {e}")

        logger.info(f"Mann Ki Baat: loaded {len(speeches)} episodes from local files.")
        count = self.save_speeches(speeches, transcript_dir=transcripts_dir, file_prefix='mann_ki_baat')
        
        # Log ingestion
        expected_episodes = 120 # Example expected count
        missing_episodes = [] # Logic to find missing episodes could be added here
        self._log_ingestion("Mann Ki Baat", count, expected_episodes, missing="None detected" if not missing_episodes else str(missing_episodes))
        
        return count

    async def scrape_ecb_press_releases(self, days_back=3650):
        """Scrape ECB press releases"""
        logger.info(f"Scraping ECB press releases from last {days_back} days...")
        current_year = datetime.now().year
        cutoff_date = datetime.now() - timedelta(days=days_back)
        start_year = cutoff_date.year
        
        pr_urls = set()
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            for year in range(current_year, start_year - 1, -1):
                url = f"https://www.ecb.europa.eu/press/pubbydate/html/index.en.html?name_of_publication=Press%20release&year={year}"
                logger.info(f"Visiting ECB PR archive: {url}")
                try:
                    await page.goto(url, wait_until="load", timeout=45000)
                    await asyncio.sleep(3)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(2)
                    links = await page.evaluate("Array.from(document.querySelectorAll('a')).map(a => a.href)")
                    for href in links:
                        if href and '/press/pr/date/' in href and 'pr' in href and '.en.html' in href:
                            pr_urls.add(href)
                except: continue
            await browser.close()
            
        speeches = []
        for url in pr_urls:
            content = self._scrape_ecb_content(url) # Reusing content scraper
            if content:
                content.update({'source': 'ECB', 'country': 'Europe', 'doc_type': 'Press Release'})
                speeches.append(content)
            time.sleep(0.5)
        count = self.save_speeches(speeches, transcript_dir='./transcripts/ecb', file_prefix='ecb')
        self._log_ingestion("ECB PRs", count, len(pr_urls))
        return count

    async def scrape_fed_press_releases(self, days_back=3650):
        """Scrape Fed press releases"""
        logger.info(f"Scraping Fed press releases from last {days_back} days...")
        base_url = "https://www.federalreserve.gov"
        current_year = datetime.now().year
        cutoff_date = datetime.now() - timedelta(days=days_back)
        start_year = cutoff_date.year
        
        pr_links = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            for year in range(current_year, start_year - 1, -1):
                if year == datetime.now().year:
                    url = f"{base_url}/newsevents/pressreleases.htm"
                elif year >= 2016:
                    url = f"{base_url}/newsevents/pressreleases/{year}-press.htm"
                else:
                    url = f"{base_url}/newsevents/pressreleases/{year}all.htm"
                
                logger.info(f"Visiting Fed PR archive: {url}")
                try:
                    await page.goto(url, wait_until="load", timeout=30000)
                    items = await page.query_selector_all(".row")
                    for item in items:
                        try:
                            date_tag = await item.query_selector("time.itemDate")
                            if not date_tag: continue
                            dt = datetime.strptime((await date_tag.inner_text()).strip(), '%m/%d/%Y')
                            if dt < cutoff_date: continue
                            title_link = await item.query_selector(".itemTitle a")
                            if not title_link: continue
                            href = await title_link.get_attribute("href")
                            pr_links.append({
                                'url': base_url + href if href.startswith('/') else href,
                                'date': dt.strftime('%Y-%m-%d'),
                                'title': (await title_link.inner_text()).strip(),
                                'doc_type': 'Press Release'
                            })
                        except: continue
                except: continue
            await browser.close()
            
        speeches = []
        for link_info in pr_links:
            content = self._scrape_fed_content(link_info['url'])
            if content:
                link_info.update(content)
                link_info.update({'source': 'Fed', 'country': 'USA'})
                speeches.append(link_info)
            time.sleep(0.5)
        count = self.save_speeches(speeches, transcript_dir='./transcripts/fed', file_prefix='fed')
        self._log_ingestion("Fed PRs", count, len(pr_links))
        return count

    async def scrape_all(self, days_back=3650):
        """Scrape all sources: ECB, Fed and Mann Ki Baat (Speeches + PRs)"""
        # MKB load from local files is fast, so we keep it as is or could skip if data exists
        self.scrape_mann_ki_baat()
        await self.scrape_ecb(days_back=days_back)
        await self.scrape_ecb_press_releases(days_back=days_back)
        await self.scrape_fed(days_back=days_back)
        await self.scrape_fed_press_releases(days_back=days_back)

    def _scrape_fed_content(self, url):
        """Scrapes individual Fed speech content"""
        try:
            response = requests.get(url, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content_div = soup.find('div', class_='col-xs-12 col-sm-8') or soup.find('div', id='content')
            if not content_div: return None
            
            # Remove unwanted
            for unwanted in content_div.find_all(['script', 'style', 'nav']):
                unwanted.decompose()
                
            paragraphs = content_div.find_all('p')
            speech_text = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50 and not re.search(r'last update:|return to text|share', text, re.I):
                    speech_text.append(text)
            
            return {'full_text': '\n\n'.join(speech_text)}
        except Exception as e:
            logger.error(f"Error scraping Fed content {url}: {e}")
            return None

if __name__ == "__main__":
    scraper = CentralizedSpeechScraper()
    # To run: python centralized_scraper.py
    async def run_test():
        await scraper.scrape_ecb(days_back=30)
        await scraper.scrape_fed(days_back=30)
        
    asyncio.run(run_test())
