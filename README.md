# Mann Ki Baat Topic Modeling Dashboard

An enhanced topic modeling application for analyzing Prime Minister Narendra Modi's "Mann Ki Baat" radio addresses using advanced NLP techniques.

## 🎯 Features

- **Advanced Topic Modeling**: Choose between NMF (Non-negative Matrix Factorization) and LDA (Latent Dirichlet Allocation)
- **Enhanced Text Preprocessing**: Part-of-speech filtering, lemmatization, and domain-specific stopword removal
- **Interactive Visualizations**: Explore topics through interactive charts and graphs using Plotly
- **Temporal Analysis**: Track how topics evolve across quarters and years
- **Episode Explorer**: Deep dive into individual episodes and their topic distributions
- **Data Export**: Download analysis results as CSV files for further research
- **Coherence Scoring**: Evaluate topic quality with automatic coherence metrics

## 📋 Prerequisites

- Python 3.8 or higher (tested on Python 3.12)
- pip package manager

## 🚀 Installation

1. **Clone the repository** (or download the files):
```bash
git clone <your-repo-url>
cd BaatSeBharat
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the spaCy language model**:
```bash
python -m spacy download en_core_web_sm
```

## 📁 Data Setup

1. Create a folder named `mann_ki_baat_transcripts` in the project directory
2. Add your transcript files with the naming pattern: `mann_ki_baat_*.txt`
3. Each file should contain:
   - Episode number and date in format: `Episode X (Date)`
   - Full transcript text

Example file structure:
```
BaatSeBharat/
├── Analysis.py
├── requirements.txt
├── README.md
└── mann_ki_baat_transcripts/
    ├── mann_ki_baat_001.txt
    ├── mann_ki_baat_002.txt
    └── ...
```

## 🎮 Usage

1. **Start the application**:
```bash
streamlit run Analysis.py
```

2. **Configure the model** (in the sidebar):
   - Choose model type (NMF recommended for cleaner topics)
   - Set number of topics (3-15, default: 8)
   - Adjust document frequency thresholds
   - Set maximum vocabulary size
   - Configure words per topic display

3. **Train the model**:
   - Click the "🚀 Train Model" button
   - Wait for preprocessing and training to complete
   - Review the topic coherence score

4. **Explore the results** across four tabs:
   - **Topic Overview**: View discovered topics with word importance charts
   - **Temporal Analysis**: See how topics evolve over time
   - **Episode Explorer**: Examine individual episodes in detail
   - **Export Data**: Download CSV files for further analysis

## 🔧 Key Parameters

### Model Configuration
- **Model Type**: 
  - NMF (Recommended): Often produces cleaner, more interpretable topics
  - LDA (Traditional): Classic probabilistic topic modeling

- **Number of Topics**: How many themes to extract (3-15)
  
- **Max Document Frequency**: Remove words appearing in >X% of documents (0.3-0.9, default: 0.5)
  
- **Min Document Count**: Remove words appearing in <X documents (2-10, default: 4)
  
- **Max Features**: Limit vocabulary size (500-2000, default: 1000)

### Text Preprocessing Features
- Part-of-speech filtering (nouns, verbs, adjectives only)
- Lemmatization using spaCy
- Extended stopword list including domain-specific terms
- Minimum token length of 3 characters
- Bigram support (1-2 word phrases)

## 📊 Topic Categories

The system automatically labels topics based on key terms:

- Healthcare & Pandemic Response
- Yoga & Wellness
- Environment & Water Conservation
- Education & Learning
- Innovation & Technology
- Social Empowerment & Youth
- Culture & Rural Development
- Agriculture & Rural Economy
- National Security & Defense
- Sports & Athletics
- General / Mixed Theme

## 🧪 Improvements Over Base Version

1. **Enhanced Preprocessing**:
   - Part-of-speech filtering reduces noise
   - Stricter token requirements (3+ characters)
   - Expanded stopword list (40+ terms)

2. **Better Vectorization**:
   - Lower max_df to remove common words
   - Higher min_df to remove rare words
   - Vocabulary size limiting

3. **NMF Support**:
   - Often produces cleaner topics than LDA
   - More interpretable word distributions
   - Faster training

4. **Weight-Based Filtering**:
   - Only displays significant terms
   - Reduces "absurd words" in topics

## 📈 Output Files

The Export tab generates two CSV files:

1. **quarterly_topics.csv**: Top 5 topics per quarter with keywords
2. **episode_topics.csv**: Topic distributions for each episode

## 🐛 Troubleshooting

**Error: "No transcripts found"**
- Check that transcript files are in `mann_ki_baat_transcripts` folder
- Verify files match the pattern `mann_ki_baat_*.txt`

**Error: "Array full of zeros"**
- Try reducing max_df or increasing min_df
- Ensure transcripts have sufficient text content
- Try switching between NMF and LDA models

**Poor topic quality (low coherence)**
- Increase min_df to remove rare words
- Decrease max_df to remove common words
- Adjust number of topics
- Try the alternative model (NMF vs LDA)

## 📚 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning (NMF, LDA, vectorization)
- **spacy**: Advanced NLP preprocessing
- **nltk**: Natural language toolkit
- **gensim**: Topic coherence calculations
- **plotly**: Interactive visualizations

## 🤝 Contributing

Suggestions for improvement:
- Additional topic labeling categories
- More visualization types
- Advanced preprocessing options
- Model comparison features

## 👥 Authors

Disha Kataria 

## 🙏 Acknowledgments

- Data source: Mann Ki Baat radio program transcripts
- Built with Streamlit and scikit-learn
- Enhanced with spaCy NLP capabilities
- Built under the guidance of Prof. Jugal Manek