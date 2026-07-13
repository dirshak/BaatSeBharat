# Leadership Rhetoric Driven Market Regime Prediction
## Ultra-Detailed Phase-Wise Implementation Plan

**Project Duration:** 30 Weeks (2 Semesters)  
**Institution:** K.J. Somaiya School of Engineering  
**Department:** Information Technology  
**Group:** 21

---

## Table of Contents

1. [Phase 0: Project Setup & Environment Configuration](#phase-0-project-setup--environment-configuration)
2. [Phase 1: Data Ingestion & Storage](#phase-1-data-ingestion--storage)
3. [Phase 2: Textual Intelligence Pipeline](#phase-2-textual-intelligence-pipeline)
4. [Phase 3: Numerical Intelligence Pipeline](#phase-3-numerical-intelligence-pipeline)
5. [Phase 4: Multimodal Fusion Engine](#phase-4-multimodal-fusion-engine)
6. [Phase 5: Validation & Testing](#phase-5-validation--testing)
7. [Phase 6: Dashboard Development](#phase-6-dashboard-development)
8. [Phase 7: Optimization & Deployment](#phase-7-optimization--deployment)
9. [Phase 8: Documentation & Publication](#phase-8-documentation--publication)

---

# Phase 0: Project Setup & Environment Configuration

**Duration:** Week 1 (5-7 days)  
**Objective:** Establish complete development environment, project structure, and version control

## Step 0.1: Development Environment Setup

### Sub-step 0.1.1: Install Base Software
**Duration:** 2 hours

1. **Install Python 3.9 or higher**
   - Download from python.org
   - Verify installation: `python --version`
   - Expected output: `Python 3.9.x` or higher
   
2. **Install Git**
   - Download from git-scm.com
   - Verify: `git --version`
   - Expected output: `git version 2.x.x`

3. **Install Visual Studio Code (or preferred IDE)**
   - Download from code.visualstudio.com
   - Install Python extension
   - Install GitLens extension

**Verification Checkpoint:**
```bash
python --version  # Should show 3.9+
git --version     # Should show git version
code --version    # Should show VS Code version
```

### Sub-step 0.1.2: Create Virtual Environment
**Duration:** 30 minutes

1. **Create project directory**
   ```bash
   mkdir leadership-market-regime
   cd leadership-market-regime
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Verify activation**
   - Prompt should show `(venv)`
   - Run: `which python` (should point to venv)

**Verification Checkpoint:**
```bash
which python  # Should point to ./venv/bin/python
pip list      # Should show minimal packages
```

### Sub-step 0.1.3: Install Core Dependencies
**Duration:** 1 hour

1. **Create requirements.txt**
   ```
   # Data Processing
   pandas==2.1.0
   numpy==1.24.3
   
   # Financial Data
   yfinance==0.2.28
   pandas-datareader==0.10.0
   
   # NLP - Embeddings
   sentence-transformers==2.2.2
   transformers==4.33.0
   torch==2.0.1
   
   # NLP - Topic Modeling
   scikit-learn==1.3.0
   gensim==4.3.2
   bertopic==0.15.0
   
   # Indian Languages
   indic-nlp-library==0.91
   
   # Statistical Analysis
   scipy==1.11.2
   statsmodels==0.14.0
   
   # Deep Learning
   pytorch-lightning==2.0.7
   
   # Visualization
   matplotlib==3.7.2
   seaborn==0.12.2
   plotly==5.16.1
   
   # Dashboard
   streamlit==1.26.0
   
   # Text Processing
   spacy==3.6.1
   nltk==3.8.1
   
   # Database
   sqlalchemy==2.0.20
   
   # Utilities
   tqdm==4.66.1
   python-dotenv==1.0.0
   requests==2.31.0
   beautifulsoup4==4.12.2
   
   # Testing
   pytest==7.4.0
   pytest-cov==4.1.0
   ```

2. **Install all packages**
   ```bash
   pip install -r requirements.txt
   ```
   - This may take 15-30 minutes
   - Watch for any errors

3. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

**Verification Checkpoint:**
```python
# Create test_imports.py
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
import spacy
import nltk
print("✓ All core imports successful")
```

**Run:** `python test_imports.py`  
**Expected:** "✓ All core imports successful"

## Step 0.2: Project Structure Creation

### Sub-step 0.2.1: Create Directory Structure
**Duration:** 30 minutes

```bash
mkdir -p data/{raw,processed,interim,external}
mkdir -p src/{data,features,models,visualization,utils}
mkdir -p notebooks/{exploratory,reports}
mkdir -p tests/{unit,integration}
mkdir -p models/{trained,checkpoints}
mkdir -p outputs/{figures,reports,exports}
mkdir -p logs
mkdir -p config
mkdir -p docs
mkdir -p scripts
```

**Verification Checkpoint:**
```bash
tree -L 2 .
# Should show all directories created
```

### Sub-step 0.2.2: Create Configuration Files
**Duration:** 1 hour

1. **Create .gitignore**
   ```
   # Python
   __pycache__/
   *.py[cod]
   *$py.class
   *.so
   .Python
   venv/
   ENV/
   
   # Data
   data/raw/*
   data/processed/*
   *.csv
   *.db
   *.sqlite
   
   # Models
   models/trained/*
   models/checkpoints/*
   
   # Outputs
   outputs/*
   logs/*
   
   # IDE
   .vscode/
   .idea/
   *.swp
   
   # OS
   .DS_Store
   Thumbs.db
   
   # Credentials
   .env
   config/secrets.yaml
   ```

2. **Create .env template**
   ```
   # API Keys (DO NOT COMMIT ACTUAL KEYS)
   FRED_API_KEY=your_key_here
   
   # Database
   DB_PATH=./data/market_rhetoric.db
   
   # Model Paths
   MODEL_DIR=./models/trained
   CHECKPOINT_DIR=./models/checkpoints
   
   # Logging
   LOG_LEVEL=INFO
   LOG_DIR=./logs
   ```

3. **Create config/config.yaml**
   ```yaml
   project:
     name: "Leadership Rhetoric Market Regime Prediction"
     version: "1.0.0"
     
   data:
     mann_ki_baat:
       source: "https://www.pmindia.gov.in/en/mann-ki-baat/"
       num_episodes: 120
       start_date: "2014-10-03"
     
     market:
       source: "yahoo_finance"
       sectors:
         - symbol: "^NSEBANK"
           name: "Banking"
         - symbol: "^CNXIT"
           name: "IT"
         - symbol: "^NSEI"
           name: "Broad Market"
       start_date: "2014-10-03"
     
   models:
     topic_modeling:
       n_topics: 20
       lda_iterations: 50
       nmf_iterations: 500
     
     counterfactual:
       input_dim: 10
       hidden_dim: 128
       num_layers: 2
       dropout: 0.2
       learning_rate: 0.001
       batch_size: 32
       epochs: 100
     
   thresholds:
     stable_regime: 0.5
     transition_regime: 1.5
     volatile_regime: 2.5
     
     warning_green: 0.3
     warning_yellow: 0.5
     warning_orange: 0.7
   ```

**Verification Checkpoint:**
- Verify all config files created
- Try loading config: `python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"`

## Step 0.3: Version Control Setup

### Sub-step 0.3.1: Initialize Git Repository
**Duration:** 30 minutes

1. **Initialize repository**
   ```bash
   git init
   ```

2. **Create initial commit**
   ```bash
   git add .gitignore requirements.txt config/
   git commit -m "Initial project setup"
   ```

3. **Create development branch**
   ```bash
   git checkout -b development
   ```

4. **Create remote repository**
   - Go to GitHub/GitLab
   - Create new repository: `leadership-market-regime`
   - Add remote: `git remote add origin <repo-url>`

5. **Push initial commit**
   ```bash
   git push -u origin main
   git push -u origin development
   ```

**Verification Checkpoint:**
```bash
git log           # Should show initial commit
git branch        # Should show main and development
git remote -v     # Should show origin
```

## Step 0.4: Database Setup

### Sub-step 0.4.1: Create SQLite Database Schema
**Duration:** 1 hour

1. **Create database initialization script**

   **File:** `scripts/init_database.py`
   ```python
   import sqlite3
   import os
   
   def create_database(db_path='./data/market_rhetoric.db'):
       """Initialize database with all required tables"""
       
       # Create directory if doesn't exist
       os.makedirs(os.path.dirname(db_path), exist_ok=True)
       
       # Connect to database
       conn = sqlite3.connect(db_path)
       cursor = conn.cursor()
       
       # Table 1: Speeches
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS speeches (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               date DATE NOT NULL,
               source TEXT NOT NULL,
               country TEXT,
               title TEXT,
               full_text TEXT,
               language TEXT,
               episode_number INTEGER,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )
       ''')
       
       # Table 2: Market Data
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS market_data (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               date DATE NOT NULL,
               ticker TEXT NOT NULL,
               sector TEXT,
               open REAL,
               high REAL,
               low REAL,
               close REAL,
               volume INTEGER,
               returns REAL,
               volatility REAL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               UNIQUE(date, ticker)
           )
       ''')
       
       # Table 3: VIX Data
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS vix_data (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               date DATE NOT NULL UNIQUE,
               vix_open REAL,
               vix_high REAL,
               vix_low REAL,
               vix_close REAL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )
       ''')
       
       # Table 4: Macro Controls
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS macro_controls (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               date DATE NOT NULL,
               indicator TEXT NOT NULL,
               value REAL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               UNIQUE(date, indicator)
           )
       ''')
       
       # Table 5: Topic Distributions
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS topic_distributions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               speech_id INTEGER NOT NULL,
               topic_id INTEGER NOT NULL,
               probability REAL NOT NULL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               FOREIGN KEY (speech_id) REFERENCES speeches(id),
               UNIQUE(speech_id, topic_id)
           )
       ''')
       
       # Table 6: Sentiment Scores
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS sentiment_scores (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               speech_id INTEGER NOT NULL,
               positive REAL,
               negative REAL,
               neutral REAL,
               compound REAL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               FOREIGN KEY (speech_id) REFERENCES speeches(id)
           )
       ''')
       
       # Table 7: Regime Classifications
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS regime_classifications (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               date DATE NOT NULL UNIQUE,
               sector TEXT NOT NULL,
               regime TEXT NOT NULL,
               confidence REAL,
               deviation_magnitude REAL,
               volume_zscore REAL,
               volatility_ratio REAL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )
       ''')
       
       # Table 8: Early Warnings
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS early_warnings (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               date DATE NOT NULL,
               sector TEXT NOT NULL,
               warning_level TEXT NOT NULL,
               warning_score REAL,
               explanation TEXT,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )
       ''')
       
       # Create indexes for performance
       cursor.execute('CREATE INDEX IF NOT EXISTS idx_speeches_date ON speeches(date)')
       cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(date)')
       cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_ticker ON market_data(ticker)')
       cursor.execute('CREATE INDEX IF NOT EXISTS idx_vix_date ON vix_data(date)')
       cursor.execute('CREATE INDEX IF NOT EXISTS idx_regime_date ON regime_classifications(date)')
       
       conn.commit()
       conn.close()
       
       print(f"✓ Database created successfully at: {db_path}")
       return db_path
   
   if __name__ == "__main__":
       create_database()
   ```

2. **Run initialization**
   ```bash
   python scripts/init_database.py
   ```

**Verification Checkpoint:**
```bash
# Check database exists
ls -lh data/market_rhetoric.db

# Verify tables
sqlite3 data/market_rhetoric.db ".tables"
# Should show: speeches, market_data, vix_data, macro_controls, 
#              topic_distributions, sentiment_scores, 
#              regime_classifications, early_warnings
```

## Step 0.5: Logging Setup

### Sub-step 0.5.1: Create Logging Configuration
**Duration:** 30 minutes

1. **Create logging utility**

   **File:** `src/utils/logger.py`
   ```python
   import logging
   import os
   from datetime import datetime
   
   def setup_logger(name, log_dir='./logs', level=logging.INFO):
       """
       Set up logger with file and console handlers
       
       Args:
           name: Logger name (usually __name__)
           log_dir: Directory for log files
           level: Logging level
       
       Returns:
           logger: Configured logger instance
       """
       # Create log directory
       os.makedirs(log_dir, exist_ok=True)
       
       # Create logger
       logger = logging.getLogger(name)
       logger.setLevel(level)
       
       # Avoid duplicate handlers
       if logger.handlers:
           return logger
       
       # Create formatters
       detailed_formatter = logging.Formatter(
           '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
           datefmt='%Y-%m-%d %H:%M:%S'
       )
       
       simple_formatter = logging.Formatter(
           '%(levelname)s: %(message)s'
       )
       
       # File handler (detailed)
       log_file = os.path.join(
           log_dir,
           f"{datetime.now().strftime('%Y%m%d')}_{name.replace('.', '_')}.log"
       )
       file_handler = logging.FileHandler(log_file)
       file_handler.setLevel(level)
       file_handler.setFormatter(detailed_formatter)
       
       # Console handler (simple)
       console_handler = logging.StreamHandler()
       console_handler.setLevel(level)
       console_handler.setFormatter(simple_formatter)
       
       # Add handlers
       logger.addHandler(file_handler)
       logger.addHandler(console_handler)
       
       return logger
   
   # Example usage
   if __name__ == "__main__":
       logger = setup_logger(__name__)
       logger.info("Logger setup complete")
       logger.debug("This is a debug message")
       logger.warning("This is a warning")
       logger.error("This is an error")
   ```

2. **Test logging**
   ```bash
   python src/utils/logger.py
   ```

**Verification Checkpoint:**
- Check `logs/` directory created
- Verify log file exists with today's date
- Open log file and verify entries

## Step 0.6: Testing Framework Setup

### Sub-step 0.6.1: Create Test Structure
**Duration:** 30 minutes

1. **Create pytest configuration**

   **File:** `pytest.ini`
   ```ini
   [pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   addopts = 
       -v
       --cov=src
       --cov-report=html
       --cov-report=term-missing
   ```

2. **Create sample test**

   **File:** `tests/test_setup.py`
   ```python
   import pytest
   import os
   import sqlite3
   
   def test_environment_setup():
       """Test that basic environment is set up correctly"""
       assert os.path.exists('./data')
       assert os.path.exists('./src')
       assert os.path.exists('./config')
   
   def test_database_exists():
       """Test that database file exists"""
       assert os.path.exists('./data/market_rhetoric.db')
   
   def test_database_tables():
       """Test that all required tables exist"""
       conn = sqlite3.connect('./data/market_rhetoric.db')
       cursor = conn.cursor()
       
       cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
       tables = [row[0] for row in cursor.fetchall()]
       
       required_tables = [
           'speeches',
           'market_data',
           'vix_data',
           'macro_controls',
           'topic_distributions',
           'sentiment_scores',
           'regime_classifications',
           'early_warnings'
       ]
       
       for table in required_tables:
           assert table in tables, f"Missing table: {table}"
       
       conn.close()
   
   def test_imports():
       """Test that all required packages can be imported"""
       import pandas
       import numpy
       import torch
       import transformers
       import spacy
       import sklearn
       import statsmodels
       assert True
   ```

3. **Run tests**
   ```bash
   pytest tests/test_setup.py -v
   ```

**Verification Checkpoint:**
- All tests should pass
- Coverage report generated in `htmlcov/`

---

# PHASE 0 COMPLETION CHECKLIST

Before proceeding to Phase 1, verify ALL of the following:

- [ ] Python 3.9+ installed and verified
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] spaCy and NLTK data downloaded
- [ ] All project directories created
- [ ] Configuration files (.gitignore, .env, config.yaml) created
- [ ] Git repository initialized
- [ ] Remote repository connected
- [ ] Database created with all 8 tables
- [ ] Database indexes created
- [ ] Logging system functional
- [ ] Test framework configured
- [ ] All setup tests passing

**Sign-off:** _______________ Date: _______________

---

# Phase 1: Data Ingestion & Storage

**Duration:** Weeks 2-3 (14 days)  
**Objective:** Collect, validate, and store all required data sources

## Step 1.1: Mann Ki Baat Data Collection

### Sub-step 1.1.1: Web Scraping Setup
**Duration:** 1 day

1. **Create scraper utility**

   **File:** `src/data/mann_ki_baat_scraper.py`
   ```python
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
   ```

2. **Test scraper (limited)**
   ```python
   # Test on 5 episodes first
   scraper = MannKiBaatScraper()
   episodes = scraper.scrape_all_episodes()[:5]
   print(f"Scraped {len(episodes)} episodes")
   ```

**Verification Checkpoint:**
```bash
python src/data/mann_ki_baat_scraper.py
# Check logs for success messages
# Verify database has entries:
sqlite3 data/market_rhetoric.db "SELECT COUNT(*) FROM speeches WHERE source='Mann Ki Baat'"
```

### Sub-step 1.1.2: Manual Data Collection (Fallback)
**Duration:** 2 days

**If web scraping fails or is blocked:**

1. **Download transcripts manually**
   - Visit PMO website
   - Download PDF/HTML versions
   - Save to `data/raw/mann_ki_baat/`

2. **Extract text from PDFs**

   **File:** `src/data/pdf_extractor.py`
   ```python
   import PyPDF2
   import os
   import glob
   
   def extract_text_from_pdf(pdf_path):
       """Extract text from PDF file"""
       with open(pdf_path, 'rb') as file:
           reader = PyPDF2.PdfReader(file)
           text = ''
           for page in reader.pages:
               text += page.extract_text()
       return text
   
   def process_all_pdfs(input_dir='./data/raw/mann_ki_baat'):
       """Process all PDFs in directory"""
       pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
       
       extracted = []
       for pdf_file in pdf_files:
           text = extract_text_from_pdf(pdf_file)
           extracted.append({
               'filename': os.path.basename(pdf_file),
               'text': text
           })
       
       return extracted
   ```

**Verification Checkpoint:**
- Verify all 120 Mann Ki Baat episodes collected
- Check text quality (readable, not garbled)
- Verify dates are correct

## Step 1.2: International Speeches Collection

### Sub-step 1.2.1: Identify Key Sources
**Duration:** 1 day

1. **Define target sources**

   **File:** `config/international_sources.yaml`
   ```yaml
   sources:
     - country: USA
       url: "https://www.whitehouse.gov/briefing-room/speeches-remarks/"
       type: economic_policy
       weight: 0.25
       
     - country: UK
       url: "https://www.gov.uk/government/speeches"
       type: economic_policy
       weight: 0.15
       
     - country: EU
       url: "https://ec.europa.eu/commission/presscorner/"
       type: economic_policy
       weight: 0.20
       
     - country: China
       url: "http://english.www.gov.cn/premier/"
       type: economic_policy
       weight: 0.20
       
     - country: UN
       url: "https://www.un.org/sg/en/content/sg/speeches"
       type: global_economic
       weight: 0.10
   ```

2. **Create scraper for international speeches**

   **File:** `src/data/international_scraper.py`
   ```python
   import requests
   from bs4 import BeautifulSoup
   import yaml
   import sqlite3
   from datetime import datetime
   import time
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
                   speech['date'],
                   'International',
                   speech['country'],
                   speech['title'],
                   speech['full_text'],
                   'English'
               ))
           
           conn.commit()
           conn.close()
   ```

**Verification Checkpoint:**
```bash
# Check international speeches collected
sqlite3 data/market_rhetoric.db "SELECT country, COUNT(*) FROM speeches WHERE source='International' GROUP BY country"
```

## Step 1.3: Market Data Collection

### Sub-step 1.3.1: Define Sectors and Tickers
**Duration:** 2 hours

1. **Create ticker configuration**

   **File:** `config/market_tickers.yaml`
   ```yaml
   sectors:
     - name: "Banking"
       tickers:
         - symbol: "^NSEBANK"
           exchange: "NSE"
           type: "index"
         - symbol: "HDFCBANK.NS"
           exchange: "NSE"
           type: "stock"
         - symbol: "ICICIBANK.NS"
           exchange: "NSE"
           type: "stock"
     
     - name: "IT"
       tickers:
         - symbol: "^CNXIT"
           exchange: "NSE"
           type: "index"
         - symbol: "TCS.NS"
           exchange: "NSE"
           type: "stock"
         - symbol: "INFY.NS"
           exchange: "NSE"
           type: "stock"
     
     - name: "Pharma"
       tickers:
         - symbol: "^CNXPHARMA"
           exchange: "NSE"
           type: "index"
         - symbol: "SUNPHARMA.NS"
           exchange: "NSE"
           type: "stock"
     
     - name: "Auto"
       tickers:
         - symbol: "^CNXAUTO"
           exchange: "NSE"
           type: "index"
         - symbol: "MARUTI.NS"
           exchange: "NSE"
           type: "stock"
     
     - name: "Energy"
       tickers:
         - symbol: "^CNXENERGY"
           exchange: "NSE"
           type: "index"
         - symbol: "RELIANCE.NS"
           exchange: "NSE"
           type: "stock"
     
     - name: "Broad Market"
       tickers:
         - symbol: "^NSEI"
           exchange: "NSE"
           type: "index"
         - symbol: "^BSESN"
           exchange: "BSE"
           type: "index"
   ```

### Sub-step 1.3.2: Download Market Data
**Duration:** 1 day

1. **Create market data downloader**

   **File:** `src/data/market_data_downloader.py`
   ```python
   import yfinance as yf
   import pandas as pd
   import yaml
   import sqlite3
   from datetime import datetime
   import time
   from src.utils.logger import setup_logger
   
   logger = setup_logger(__name__)
   
   class MarketDataDownloader:
       """Download market data from Yahoo Finance"""
       
       def __init__(self, config_path='./config/market_tickers.yaml'):
           with open(config_path) as f:
               config = yaml.safe_load(f)
           self.sectors = config['sectors']
       
       def download_ticker_data(self, ticker_info, sector_name, 
                               start_date='2014-10-03'):
           """Download data for a single ticker"""
           symbol = ticker_info['symbol']
           
           logger.info(f"Downloading {symbol} ({sector_name})...")
           
           try:
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
           
           conn = sqlite3.connect(db_path)
           
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
   ```

2. **Run downloader**
   ```bash
   python src/data/market_data_downloader.py
   ```

**Verification Checkpoint:**
```bash
# Check market data
sqlite3 data/market_rhetoric.db "SELECT sector, ticker, COUNT(*) as records FROM market_data GROUP BY sector, ticker"

# Should show records for all tickers
# Verify date range
sqlite3 data/market_rhetoric.db "SELECT MIN(date), MAX(date) FROM market_data"
```

## Step 1.4: VIX Data Collection

### Sub-step 1.4.1: Download VIX Data
**Duration:** 2 hours

1. **Create VIX downloader**

   **File:** `src/data/vix_downloader.py`
   ```python
   import yfinance as yf
   import sqlite3
   from datetime import datetime
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
   ```

2. **Run VIX downloader**
   ```bash
   python src/data/vix_downloader.py
   ```

**Verification Checkpoint:**
```bash
# Check VIX data
sqlite3 data/market_rhetoric.db "SELECT COUNT(*) FROM vix_data"
sqlite3 data/market_rhetoric.db "SELECT MIN(date), MAX(date), AVG(vix_close) FROM vix_data"
```

## Step 1.5: Data Quality Validation

### Sub-step 1.5.1: Create Validation Tests
**Duration:** 1 day

1. **Create data validation script**

   **File:** `tests/test_data_quality.py`
   ```python
   import pytest
   import sqlite3
   import pandas as pd
   from datetime import datetime
   
   @pytest.fixture
   def db_connection():
       """Database connection fixture"""
       conn = sqlite3.connect('./data/market_rhetoric.db')
       yield conn
       conn.close()
   
   def test_mann_ki_baat_count(db_connection):
       """Test that we have sufficient Mann Ki Baat episodes"""
       cursor = db_connection.cursor()
       cursor.execute("SELECT COUNT(*) FROM speeches WHERE source='Mann Ki Baat'")
       count = cursor.fetchone()[0]
       
       assert count >= 100, f"Expected at least 100 episodes, got {count}"
   
   def test_mann_ki_baat_dates(db_connection):
       """Test that Mann Ki Baat dates are valid"""
       df = pd.read_sql_query(
           "SELECT date FROM speeches WHERE source='Mann Ki Baat'",
           db_connection
       )
       
       df['date'] = pd.to_datetime(df['date'])
       
       # Check date range
       assert df['date'].min() >= pd.Timestamp('2014-10-01'), "Start date too early"
       assert df['date'].max() <= pd.Timestamp.now(), "Future dates found"
       
       # Check for nulls
       assert df['date'].notna().all(), "Null dates found"
   
   def test_market_data_completeness(db_connection):
       """Test market data completeness"""
       df = pd.read_sql_query(
           "SELECT ticker, COUNT(*) as count FROM market_data GROUP BY ticker",
           db_connection
       )
       
       # Each ticker should have at least 2000 records (roughly 10 years of trading days)
       assert (df['count'] >= 2000).all(), "Insufficient market data for some tickers"
   
   def test_market_data_quality(db_connection):
       """Test market data quality"""
       df = pd.read_sql_query(
           "SELECT * FROM market_data LIMIT 1000",
           db_connection
       )
       
       # Check for negative prices
       assert (df['close'] > 0).all(), "Negative prices found"
       assert (df['high'] >= df['low']).all(), "High < Low found"
       assert (df['high'] >= df['close']).all(), "High < Close found"
       assert (df['low'] <= df['close']).all(), "Low > Close found"
       
       # Check for extreme values (likely errors)
       assert (df['close'] < df['close'].mean() * 10).all(), "Extreme price values found"
   
   def test_vix_data_exists(db_connection):
       """Test VIX data exists"""
       cursor = db_connection.cursor()
       cursor.execute("SELECT COUNT(*) FROM vix_data")
       count = cursor.fetchone()[0]
       
       assert count >= 2000, f"Insufficient VIX data: {count} records"
   
   def test_vix_data_range(db_connection):
       """Test VIX values are in reasonable range"""
       df = pd.read_sql_query("SELECT vix_close FROM vix_data", db_connection)
       
       # VIX typically ranges from 10-80, rarely exceeds 100
       assert (df['vix_close'] >= 5).all(), "VIX too low (likely error)"
       assert (df['vix_close'] <= 100).all(), "VIX too high (likely error)"
   
   def test_no_duplicate_dates_per_ticker(db_connection):
       """Test no duplicate date-ticker combinations"""
       cursor = db_connection.cursor()
       cursor.execute('''
           SELECT date, ticker, COUNT(*) as count
           FROM market_data
           GROUP BY date, ticker
           HAVING count > 1
       ''')
       
       duplicates = cursor.fetchall()
       assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate date-ticker combinations"
   
   def test_speech_text_not_empty(db_connection):
       """Test that speeches have actual text content"""
       df = pd.read_sql_query("SELECT full_text FROM speeches", db_connection)
       
       # Check that text is not empty or too short
       text_lengths = df['full_text'].str.len()
       assert (text_lengths > 100).all(), "Some speeches have very short or empty text"
   ```

2. **Run validation tests**
   ```bash
   pytest tests/test_data_quality.py -v
   ```

**Verification Checkpoint:**
- ALL validation tests must pass
- If any test fails, investigate and fix data issues
- Re-run tests until all pass

---

# PHASE 1 COMPLETION CHECKLIST

- [ ] Mann Ki Baat: 100+ episodes collected
- [ ] Mann Ki Baat: All dates valid and in correct range
- [ ] Mann Ki Baat: Text content not empty
- [ ] International: 50+ speeches collected from 5+ countries
- [ ] Market Data: All tickers have 2000+ records
- [ ] Market Data: No negative prices or invalid values
- [ ] Market Data: Date range covers 2014-10-03 to present
- [ ] VIX Data: 2000+ records collected
- [ ] VIX Data: Values in reasonable range (5-100)
- [ ] Database: No duplicate entries
- [ ] Database: All foreign key relationships intact
- [ ] Validation: All data quality tests passing

**Sign-off:** _______________ Date: _______________

---

# Phase 2: Textual Intelligence Pipeline

**Duration:** Weeks 4-7 (28 days)  
**Objective:** Extract topics and sentiment from speeches

## Step 2.1: Text Preprocessing

### Sub-step 2.1.1: Create Preprocessing Utilities
**Duration:** 2 days

1. **Create text cleaning module**

   **File:** `src/features/text_preprocessing.py`
   ```python
   import spacy
   import re
   from indicnlp.tokenize import indic_tokenize
   from indicnlp.normalize import indic_normalize
   import pandas as pd
   from src.utils.logger import setup_logger
   
   logger = setup_logger(__name__)
   
   class TextPreprocessor:
       """
       Comprehensive text preprocessing for multilingual speeches
       """
       
       def __init__(self):
           # Load English model
           try:
               self.nlp_en = spacy.load('en_core_web_sm')
           except:
               logger.error("English spaCy model not found. Run: python -m spacy download en_core_web_sm")
               raise
           
           # Hindi normalizer
           self.hindi_normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("hi")
           
           # Extended stopwords
           self.custom_stopwords = {
               'mann', 'ki', 'baat', 'friends', 'dear', 'countrymen',
               'namaskar', 'namaste', 'ji', 'today', 'time', 'country',
               'people', 'india', 'indian', 'government', 'year', 'month',
               'day', 'week', 'says', 'said', 'will', 'also', 'many', 'much',
               'make', 'made', 'take', 'taken', 'get', 'got', 'give', 'given'
           }
       
       def clean_text(self, text):
           """Basic text cleaning"""
           if not text:
               return ""
           
           # Convert to lowercase
           text = text.lower()
           
           # Remove URLs
           text = re.sub(r'http\S+|www\S+', '', text)
           
           # Remove email addresses
           text = re.sub(r'\S+@\S+', '', text)
           
           # Remove special characters but keep spaces
           text = re.sub(r'[^\w\s]', ' ', text)
           
           # Remove extra whitespace
           text = re.sub(r'\s+', ' ', text).strip()
           
           return text
       
       def preprocess_english(self, text):
           """
           Preprocess English text with POS filtering and lemmatization
           """
           # Clean text
           text = self.clean_text(text)
           
           # Process with spaCy
           doc = self.nlp_en(text)
           
           # POS filtering and lemmatization
           filtered_tokens = []
           
           for token in doc:
               # Filter conditions
               if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and
                   not token.is_stop and
                   not token.is_punct and
                   len(token.text) > 2 and
                   token.lemma_ not in self.custom_stopwords):
                   
                   filtered_tokens.append(token.lemma_)
           
           return ' '.join(filtered_tokens)
       
       def preprocess_hindi(self, text):
           """
           Preprocess Hindi text
           """
           # Normalize
           normalized = self.hindi_normalizer.normalize(text)
           
           # Tokenize
           tokens = indic_tokenize.trivial_tokenize(normalized)
           
           # Filter
           filtered = [
               t for t in tokens 
               if len(t) > 1 and t.isalnum()
           ]
           
           return ' '.join(filtered)
       
       def detect_language(self, text):
           """
           Simple language detection (Hindi vs English)
           """
           # Check for Devanagari script
           devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
           
           if devanagari_chars > len(text) * 0.3:
               return 'hindi'
           else:
               return 'english'
       
       def preprocess(self, text, language=None):
           """
           Automatic preprocessing based on detected language
           """
           if not language:
               language = self.detect_language(text)
           
           if language == 'hindi':
               return self.preprocess_hindi(text)
           else:
               return self.preprocess_english(text)
       
       def segment_sentences(self, text, language='english'):
           """
           Segment text into sentences
           Important for handling BERT's 512 token limit
           """
           if language == 'english':
               doc = self.nlp_en(text)
               return [sent.text for sent in doc.sents]
           else:
               # Hindi sentence segmentation
               sentences = text.split('।')  # Devanagari full stop
               return [s.strip() for s in sentences if s.strip()]
       
       def create_hierarchical_structure(self, text, language='english'):
           """
           Create Episode > Paragraph > Sentence hierarchy
           """
           paragraphs = text.split('\n\n')
           
           hierarchy = {
               'episode_text': text,
               'paragraphs': []
           }
           
           for para in paragraphs:
               if para.strip():
                   sentences = self.segment_sentences(para, language)
                   hierarchy['paragraphs'].append({
                       'text': para,
                       'sentences': sentences,
                       'n_sentences': len(sentences)
                   })
           
           return hierarchy
   
   # Testing
   if __name__ == "__main__":
       preprocessor = TextPreprocessor()
       
       # Test English
       test_text_en = "Friends, today I want to talk about India's progress in technology."
       processed = preprocessor.preprocess_english(test_text_en)
       print(f"Original: {test_text_en}")
       print(f"Processed: {processed}")
       
       # Test hierarchy
       long_text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
       hierarchy = preprocessor.create_hierarchical_structure(long_text)
       print(f"Paragraphs: {len(hierarchy['paragraphs'])}")
   ```

2. **Test preprocessing**
   ```bash
   python src/features/text_preprocessing.py
   ```

**Verification Checkpoint:**
- Verify English preprocessing works correctly
- Check POS filtering removes correct words
- Verify lemmatization (e.g., "running" → "run")
- Test hierarchy creation

### Sub-step 2.1.2: Batch Preprocess All Speeches
**Duration:** 1 day

1. **Create batch processing script**

   **File:** `scripts/preprocess_all_speeches.py`
   ```python
   import sqlite3
   import pandas as pd
   from tqdm import tqdm
   import sys
   import os
   
   sys.path.append(os.path.dirname(os.path.dirname(__file__)))
   from src.features.text_preprocessing import TextPreprocessor
   from src.utils.logger import setup_logger
   
   logger = setup_logger(__name__)
   
   def preprocess_all_speeches(db_path='./data/market_rhetoric.db'):
       """
       Preprocess all speeches in database
       """
       logger.info("Loading speeches from database...")
       
       # Load speeches
       conn = sqlite3.connect(db_path)
       df = pd.read_sql_query("SELECT id, full_text FROM speeches", conn)
       
       logger.info(f"Loaded {len(df)} speeches")
       
       # Initialize preprocessor
       preprocessor = TextPreprocessor()
       
       # Process each speech
       logger.info("Preprocessing speeches...")
       
       processed_texts = []
       
       for idx, row in tqdm(df.iterrows(), total=len(df)):
           try:
               # Preprocess
               processed = preprocessor.preprocess(row['full_text'])
               processed_texts.append(processed)
           except Exception as e:
               logger.error(f"Error processing speech {row['id']}: {e}")
               processed_texts.append("")
       
       # Add to dataframe
       df['processed_text'] = processed_texts
       
       # Save processed text back to database
       logger.info("Saving processed text to database...")
       
       # Add column if doesn't exist
       conn.execute('''
           ALTER TABLE speeches 
           ADD COLUMN processed_text TEXT
       ''')
       
       # Update each row
       for idx, row in df.iterrows():
           conn.execute(
               "UPDATE speeches SET processed_text = ? WHERE id = ?",
               (row['processed_text'], row['id'])
           )
       
       conn.commit()
       conn.close()
       
       logger.info("✓ Preprocessing complete")
       
       # Statistics
       avg_length = df['processed_text'].str.len().mean()
       logger.info(f"Average processed text length: {avg_length:.0f} characters")
   
   if __name__ == "__main__":
       preprocess_all_speeches()
   ```

2. **Run batch preprocessing**
   ```bash
   python scripts/preprocess_all_speeches.py
   ```

**Verification Checkpoint:**
```bash
# Check processed text exists
sqlite3 data/market_rhetoric.db "SELECT COUNT(*) FROM speeches WHERE processed_text IS NOT NULL AND processed_text != ''"

# Check average length
sqlite3 data/market_rhetoric.db "SELECT AVG(LENGTH(processed_text)) FROM speeches"
```

## Step 2.2: Text Embeddings

### Sub-step 2.2.1: Generate Sentence Embeddings
**Duration:** 2 days

1. **Create embedding generator**

   **File:** `src/features/text_embeddings.py`
   ```python
   from sentence_transformers import SentenceTransformer
   from transformers import AutoTokenizer, AutoModel
   import torch
   import numpy as np
   import pandas as pd
   import sqlite3
   from tqdm import tqdm
   from src.utils.logger import setup_logger
   
   logger = setup_logger(__name__)
   
   class TextEmbeddingGenerator:
       """
       Generate text embeddings using SBERT and IndicBERT
       """
       
       def __init__(self):
           logger.info("Loading embedding models...")
           
           # SBERT for English (384-dim)
           self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
           
           # IndicBERT for Hindi (768-dim)
           self.indic_tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
           self.indic_model = AutoModel.from_pretrained('ai4bharat/indic-bert')
           self.indic_model.eval()
           
           logger.info("✓ Models loaded successfully")
       
       def embed_sbert(self, texts, batch_size=32, show_progress=True):
           """
           Generate SBERT embeddings
           Returns: numpy array of shape (n_texts, 384)
           """
           logger.info(f"Generating SBERT embeddings for {len(texts)} texts...")
           
           embeddings = self.sbert_model.encode(
               texts,
               batch_size=batch_size,
               show_progress_bar=show_progress,
               convert_to_numpy=True
           )
           
           logger.info(f"✓ Generated embeddings with shape {embeddings.shape}")
           return embeddings
       
       def embed_indicbert(self, texts, batch_size=16, show_progress=True):
           """
           Generate IndicBERT embeddings for Hindi
           Returns: numpy array of shape (n_texts, 768)
           """
           logger.info(f"Generating IndicBERT embeddings for {len(texts)} texts...")
           
           all_embeddings = []
           
           # Process in batches
           for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress):
               batch = texts[i:i+batch_size]
               
               # Tokenize
               encoded = self.indic_tokenizer(
                   batch,
                   padding=True,
                   truncation=True,
                   max_length=512,
                   return_tensors='pt'
               )
               
               # Generate embeddings
               with torch.no_grad():
                   outputs = self.indic_model(**encoded)
                   # Use CLS token
                   batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
               
               all_embeddings.append(batch_embeddings)
           
           embeddings = np.vstack(all_embeddings)
           logger.info(f"✓ Generated embeddings with shape {embeddings.shape}")
           
           return embeddings
       
       def generate_and_save_embeddings(self, db_path='./data/market_rhetoric.db'):
           """
           Generate embeddings for all speeches and save
           """
           # Load speeches
           conn = sqlite3.connect(db_path)
           df = pd.read_sql_query(
               "SELECT id, processed_text, language FROM speeches WHERE processed_text IS NOT NULL",
               conn
           )
           
           logger.info(f"Generating embeddings for {len(df)} speeches...")
           
           # Separate by language
           english_df = df[df['language'].str.contains('English', case=False, na=False)]
           hindi_df = df[~df['language'].str.contains('English', case=False, na=False)]
           
           # Generate embeddings
           embeddings_dict = {}
           
           if len(english_df) > 0:
               logger.info(f"Processing {len(english_df)} English speeches...")
               english_embeddings = self.embed_sbert(english_df['processed_text'].tolist())
               
               for idx, speech_id in enumerate(english_df['id']):
                   embeddings_dict[speech_id] = english_embeddings[idx]
           
           if len(hindi_df) > 0:
               logger.info(f"Processing {len(hindi_df)} Hindi speeches...")
               hindi_embeddings = self.embed_indicbert(hindi_df['processed_text'].tolist())
               
               for idx, speech_id in enumerate(hindi_df['id']):
                   embeddings_dict[speech_id] = hindi_embeddings[idx]
           
           # Save embeddings
           logger.info("Saving embeddings...")
           
           # Save as numpy file
           np.save('./data/processed/speech_embeddings.npy', embeddings_dict)
           
           # Also save mapping
           import pickle
           with open('./data/processed/speech_id_mapping.pkl', 'wb') as f:
               pickle.dump(list(embeddings_dict.keys()), f)
           
           conn.close()
           
           logger.info(f"✓ Saved embeddings for {len(embeddings_dict)} speeches")
           
           return embeddings_dict
   
   if __name__ == "__main__":
       generator = TextEmbeddingGenerator()
       embeddings = generator.generate_and_save_embeddings()
   ```

2. **Generate embeddings**
   ```bash
   python src/features/text_embeddings.py
   ```

   **Note:** This may take 1-2 hours depending on number of speeches and hardware

**Verification Checkpoint:**
```python
# Verify embeddings saved
import numpy as np
embeddings = np.load('./data/processed/speech_embeddings.npy', allow_pickle=True).item()
print(f"Embeddings for {len(embeddings)} speeches")
print(f"Embedding dimension: {list(embeddings.values())[0].shape}")
```

## Step 2.3: Topic Modeling

### Sub-step 2.3.1: Train Ensemble Topic Models
**Duration:** 3 days

1. **Create topic modeling module**

   **File:** `src/models/topic_modeling.py`
   ```python
   from sklearn.decomposition import LatentDirichletAllocation, NMF
   from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
   from bertopic import BERTopic
   import numpy as np
   import pandas as pd
   import sqlite3
   import pickle
   from src.utils.logger import setup_logger
   
   logger = setup_logger(__name__)
   
   class HybridTopicModeler:
       """
       Ensemble topic modeling: LDA + NMF + BERTopic
       """
       
       def __init__(self, n_topics=20):
           self.n_topics = n_topics
           
           # Models
           self.lda_model = None
           self.nmf_model = None
           self.bertopic_model = None
           
           # Vectorizers
           self.count_vectorizer = None
           self.tfidf_vectorizer = None
       
       def fit_lda(self, documents):
           """Fit LDA model"""
           logger.info("Fitting LDA model...")
           
           # Create vectorizer
           self.count_vectorizer = CountVectorizer(
               max_features=5000,
               min_df=2,
               max_df=0.8,
               stop_words='english'
           )
           
           doc_term_matrix = self.count_vectorizer.fit_transform(documents)
           
           # Train LDA
           self.lda_model = LatentDirichletAllocation(
               n_components=self.n_topics,
               random_state=42,
               max_iter=50,
               learning_method='batch',
               n_jobs=-1
           )
           
           self.lda_model.fit(doc_term_matrix)
           
           logger.info("✓ LDA model fitted")
           
           return self.lda_model.transform(doc_term_matrix)
       
       def fit_nmf(self, documents):
           """Fit NMF model"""
           logger.info("Fitting NMF model...")
           
           # Create vectorizer
           self.tfidf_vectorizer = TfidfVectorizer(
               max_features=5000,
               min_df=2,
               max_df=0.8,
               stop_words='english'
           )
           
           tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
           
           # Train NMF
           self.nmf_model = NMF(
               n_components=self.n_topics,
               random_state=42,
               max_iter=500,
               init='nndsvda'
           )
           
           self.nmf_model.fit(tfidf_matrix)
           
           logger.info("✓ NMF model fitted")
           
           return self.nmf_model.transform(tfidf_matrix)
       
       def fit_bertopic(self, documents, embeddings):
           """Fit BERTopic model"""
           logger.info("Fitting BERTopic model...")
           
           # Initialize BERTopic
           self.bertopic_model = BERTopic(
               nr_topics=self.n_topics,
               calculate_probabilities=True,
               verbose=False
           )
           
           # Fit
           topics, probs = self.bertopic_model.fit_transform(documents, embeddings)
           
           logger.info("✓ BERTopic model fitted")
           
           # Convert to topic distributions
           topic_dist = np.zeros((len(documents), self.n_topics))
           for i, (topic, prob_dist) in enumerate(zip(topics, probs)):
               if topic != -1:
                   topic_dist[i] = prob_dist
           
           return topic_dist
       
       def fit_ensemble(self, documents, embeddings):
           """
           Fit all three models
           """
           logger.info(f"Training ensemble on {len(documents)} documents...")
           
           # Fit each model
           lda_topics = self.fit_lda(documents)
           nmf_topics = self.fit_nmf(documents)
           bert_topics = self.fit_bertopic(documents, embeddings)
           
           distributions = {
               'lda': lda_topics,
               'nmf': nmf_topics,
               'bertopic': bert_topics
           }
           
           # Create consensus
           consensus = self.create_consensus(distributions)
           
           logger.info("✓ Ensemble training complete")
           
           return consensus, distributions
       
       def create_consensus(self, distributions):
           """
           Weighted ensemble of topic distributions
           """
           consensus = (
               0.3 * distributions['lda'] +
               0.3 * distributions['nmf'] +
               0.4 * distributions['bertopic']
           )
           
           # Normalize
           consensus = consensus / consensus.sum(axis=1, keepdims=True)
           
           return consensus
       
       def get_topic_labels(self):
           """
           Extract topic labels from each model
           """
           labels = {}
           
           # LDA topics
           if self.lda_model and self.count_vectorizer:
               feature_names = self.count_vectorizer.get_feature_names_out()
               for topic_idx, topic in enumerate(self.lda_model.components_):
                   top_words_idx = topic.argsort()[-10:][::-1]
                   top_words = [feature_names[i] for i in top_words_idx]
                   labels[f'Topic_{topic_idx}'] = {
                       'model': 'LDA',
                       'keywords': top_words
                   }
           
           # BERTopic topics
           if self.bertopic_model:
               topic_info = self.bertopic_model.get_topic_info()
               for idx, row in topic_info.iterrows():
                   if row['Topic'] != -1:
                       labels[f'Topic_{row["Topic"]}']['bertopic_label'] = row['Name']
           
           return labels
       
       def save_models(self, output_dir='./models/trained'):
           """Save all models"""
           import os
           os.makedirs(output_dir, exist_ok=True)
           
           # Save each model
           if self.lda_model:
               with open(f'{output_dir}/lda_model.pkl', 'wb') as f:
                   pickle.dump(self.lda_model, f)
               with open(f'{output_dir}/count_vectorizer.pkl', 'wb') as f:
                   pickle.dump(self.count_vectorizer, f)
           
           if self.nmf_model:
               with open(f'{output_dir}/nmf_model.pkl', 'wb') as f:
                   pickle.dump(self.nmf_model, f)
               with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
                   pickle.dump(self.tfidf_vectorizer, f)
           
           if self.bertopic_model:
               with open(f'{output_dir}/bertopic_model.pkl', 'wb') as f:
                   pickle.dump(self.bertopic_model, f)
           
           logger.info(f"✓ Models saved to {output_dir}")
   
   if __name__ == "__main__":
       # Load data
       conn = sqlite3.connect('./data/market_rhetoric.db')
       df = pd.read_sql_query(
           "SELECT processed_text FROM speeches WHERE processed_text IS NOT NULL",
           conn
       )
       conn.close()
       
       documents = df['processed_text'].tolist()
       
       # Load embeddings
       embeddings_dict = np.load('./data/processed/speech_embeddings.npy', allow_pickle=True).item()
       embeddings = np.array(list(embeddings_dict.values()))
       
       # Train models
       modeler = HybridTopicModeler(n_topics=20)
       consensus, distributions = modeler.fit_ensemble(documents, embeddings)
       
       # Save
       modeler.save_models()
       np.save('./data/processed/topic_distributions.npy', consensus)
   ```

2. **Train topic models**
   ```bash
   python src/models/topic_modeling.py
   ```

   **Note:** This may take 2-4 hours

**Verification Checkpoint:**
```python
# Check models saved
import os
assert os.path.exists('./models/trained/lda_model.pkl')
assert os.path.exists('./models/trained/nmf_model.pkl')
assert os.path.exists('./models/trained/bertopic_model.pkl')

# Check topic distributions
import numpy as np
topics = np.load('./data/processed/topic_distributions.npy')
print(f"Topic distributions shape: {topics.shape}")
print(f"Sum of probabilities per doc: {topics.sum(axis=1)[:5]}")  # Should be close to 1
```

### Sub-step 2.3.2: Calculate Topic Coherence
**Duration:** 1 day

1. **Create coherence calculator**

   **File:** `src/models/topic_coherence.py`
   ```python
   from gensim.models.coherencemodel import CoherenceModel
   from gensim.corpora import Dictionary
   import pandas as pd
   import sqlite3
   import pickle
   from src.utils.logger import setup_logger
   
   logger = setup_logger(__name__)
   
   def calculate_coherence(model_path, documents):
       """
       Calculate topic coherence for validation
       """
       logger.info("Calculating topic coherence...")
       
       # Load model
       with open(model_path, 'rb') as f:
           model = pickle.load(f)
       
       # Tokenize documents
       texts = [doc.split() for doc in documents]
       
       # Create dictionary
       dictionary = Dictionary(texts)
       
       # Calculate coherence
       try:
           coherence_model = CoherenceModel(
               model=model,
               texts=texts,
               dictionary=dictionary,
               coherence='c_v'
           )
           
           coherence_score = coherence_model.get_coherence()
           
           logger.info(f"✓ Coherence score: {coherence_score:.4f}")
           
           return coherence_score
           
       except Exception as e:
           logger.error(f"Error calculating coherence: {e}")
           return None
   
   if __name__ == "__main__":
       # Load documents
       conn = sqlite3.connect('./data/market_rhetoric.db')
       df = pd.read_sql_query(
           "SELECT processed_text FROM speeches WHERE processed_text IS NOT NULL",
           conn
       )
       conn.close()
       
       documents = df['processed_text'].tolist()
       
       # Calculate for each model
       lda_coherence = calculate_coherence('./models/trained/lda_model.pkl', documents)
       print(f"LDA Coherence: {lda_coherence}")
   ```

2. **Run coherence calculation**
   ```bash
   python src/models/topic_coherence.py
   ```

**Verification Checkpoint:**
- Coherence score should be > 0.3 (acceptable)
- Coherence score > 0.5 is good
- If coherence < 0.3, consider re-training with different parameters

---

**Due to character limits, I'll continue this in the next response. This implementation plan is extremely detailed and will span approximately 100-150 pages when complete.**

**Would you like me to:**
1. Continue with the remaining phases (Sentiment Analysis, Numerical Intelligence, Fusion, Validation, Dashboard, Optimization, Documentation)?
2. Focus on a specific phase you'd like more detail on?
3. Provide this first portion as a downloadable file?

The complete plan will include all phases with this level of granular detail including:
- Every sub-step with duration estimates
- Complete code implementations
- Verification checkpoints after each step
- Error handling procedures
- Testing strategies
- Sign-off checklists

Please let me know how you'd like to proceed!