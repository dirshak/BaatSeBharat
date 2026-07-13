"""
Groq-hosted Llama classifier: reads a speech's dominant NMF topic (label +
keywords) plus a text excerpt, and decides which companies in
prediction_engine.COMPANY_UNIVERSE it's actually relevant to, with a
strength (strong/weak) and sentiment (positive/negative/neutral) judgment
per company.

This does NOT replace the NMF topic model (src/models/topic_modeling.py) or
FinBERT sentiment (src/models/sentiment_overlay.py) -- it adds a
content-aware company mapping neither of those produces on their own. The
output is stored in `llm_company_signals` and blended into
prediction_engine.get_company_prediction() as an additional weighted input
(config/config.yaml -> models.groq_classifier.blend_weight).

The company universe Groq is allowed to choose from is intentionally
restricted to prediction_engine.COMPANY_UNIVERSE -- it must never invent a
ticker the rest of the pipeline (yfinance fetches, beta profiles) doesn't
already know about.
"""

import json
import os
import sys
import time

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger

try:
    from prediction_engine import COMPANY_UNIVERSE, SECTOR_COMPANIES, get_company_sector_by_ticker
except ImportError:
    from src.prediction_engine import COMPANY_UNIVERSE, SECTOR_COMPANIES, get_company_sector_by_ticker

logger = setup_logger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'config', 'config.yaml'
)

_VALID_STRENGTHS = {'strong', 'weak'}
_VALID_SENTIMENTS = {'positive', 'negative', 'neutral'}


def load_groq_config(path=DEFAULT_CONFIG_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('models', {}).get('groq_classifier', {}) or {}


def _company_universe_prompt_block():
    """Format the fixed company universe as `Name (TICKER) — Sector` lines,
    the only companies the LLM is allowed to reference."""
    lines = []
    for name, ticker in COMPANY_UNIVERSE.items():
        sector = get_company_sector_by_ticker(ticker)
        lines.append(f"- {name} ({ticker}) — {sector}")
    return "\n".join(lines)


_SYSTEM_PROMPT = """You are a financial analyst classifying how a leadership \
speech's topic affects specific Indian-listed companies.

You will be given: the speech's dominant topic (label + top keywords, from \
a statistical topic model -- treat these as context, not ground truth), and \
an excerpt of the speech text. You must decide which companies from the \
provided list are genuinely, materially relevant to this topic -- most \
speeches are relevant to few or zero companies. Do not force a mapping.

For each relevant company, output:
  - "ticker": exactly one of the tickers from the provided list
  - "strength": "strong" or "weak" -- how directly/materially this topic \
affects that company
  - "sentiment": "positive", "negative", or "neutral" -- the directional \
tone of the topic for that company
  - "confidence": a number from 0.0 to 1.0
  - "rationale": one short sentence

Respond with ONLY a JSON object of the form:
{"companies": [{"ticker": "...", "strength": "...", "sentiment": "...", "confidence": 0.0, "rationale": "..."}]}
If no company is relevant, respond {"companies": []}. Never invent a ticker \
that isn't in the provided list."""


class GroqTopicClassifier:
    """Wraps the Groq chat-completions API for speech -> company
    classification. Raises on missing API key / SDK; callers (the batch
    script) decide how to handle that at the process level."""

    def __init__(self, model=None, api_key=None, config_path=DEFAULT_CONFIG_PATH):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is required for LLM topic classification. "
                "Install it with: pip install groq"
            ) from exc

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set (checked argument and environment).")

        self.cfg = load_groq_config(config_path)
        self.model = model or self.cfg.get('model', 'llama-3.3-70b-versatile')
        self.temperature = float(self.cfg.get('temperature', 0.1))
        self.max_output_tokens = int(self.cfg.get('max_output_tokens', 600))
        self.strength_scores = self.cfg.get('strength_scores', {'strong': 0.85, 'weak': 0.3})
        self.sentiment_scores = self.cfg.get(
            'sentiment_scores', {'positive': 0.7, 'negative': -0.7, 'neutral': 0.0}
        )

        self.client = Groq(api_key=api_key)
        self._company_block = _company_universe_prompt_block()
        self._valid_tickers = set(COMPANY_UNIVERSE.values())

    def _build_user_prompt(self, text_excerpt, topic_label, topic_keywords):
        keywords_str = ", ".join(topic_keywords[:15]) if topic_keywords else "(none)"
        excerpt = (text_excerpt or "")[:3000]
        return (
            f"Dominant topic label: {topic_label or '(unlabeled)'}\n"
            f"Topic keywords: {keywords_str}\n\n"
            f"Companies you may reference (ticker in parentheses):\n{self._company_block}\n\n"
            f"Speech excerpt:\n{excerpt}"
        )

    def classify_speech(self, text_excerpt, topic_label=None, topic_keywords=None, max_retries=3):
        """Returns a list of validated {ticker, strength, sentiment,
        confidence, rationale} dicts. Never raises on malformed model
        output -- logs and returns [] instead, so a single bad response
        doesn't kill a batch run."""
        user_prompt = self._build_user_prompt(text_excerpt, topic_label, topic_keywords)

        last_exc = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw = response.choices[0].message.content
                return self._parse_and_validate(raw), raw
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Groq classification attempt %d/%d failed: %s (retrying in %ds)",
                    attempt + 1, max_retries, exc, wait
                )
                time.sleep(wait)

        logger.error("Groq classification failed after %d attempts: %s", max_retries, last_exc)
        return [], None

    def _parse_and_validate(self, raw):
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Groq returned non-JSON output, skipping: %s", exc)
            return []

        items = payload.get('companies', []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            logger.warning("Groq 'companies' field was not a list, skipping.")
            return []

        valid = []
        for item in items:
            if not isinstance(item, dict):
                continue
            ticker = item.get('ticker')
            strength = item.get('strength')
            sentiment = item.get('sentiment')
            if ticker not in self._valid_tickers:
                logger.warning("Groq referenced unknown ticker %r, dropping.", ticker)
                continue
            if strength not in _VALID_STRENGTHS or sentiment not in _VALID_SENTIMENTS:
                logger.warning(
                    "Groq gave invalid strength/sentiment for %s (%r/%r), dropping.",
                    ticker, strength, sentiment
                )
                continue
            try:
                confidence = float(item.get('confidence', 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            valid.append({
                'ticker': ticker,
                'strength': strength,
                'sentiment': sentiment,
                'confidence': max(0.0, min(1.0, confidence)),
                'rationale': str(item.get('rationale', ''))[:500],
            })
        return valid

    def classify_and_store(self, conn, speech_id, text_excerpt, topic_label=None, topic_keywords=None):
        """Classifies one speech and upserts results into
        llm_company_signals. Returns the number of company rows stored."""
        companies, raw = self.classify_speech(text_excerpt, topic_label, topic_keywords)

        ticker_to_name = {t: n for n, t in COMPANY_UNIVERSE.items()}
        stored = 0
        for item in companies:
            ticker = item['ticker']
            strength_score = float(self.strength_scores.get(item['strength'], 0.5))
            sentiment_score = float(self.sentiment_scores.get(item['sentiment'], 0.0))
            conn.execute('''
                INSERT INTO llm_company_signals
                    (speech_id, ticker, company_name, topic_label, strength, sentiment,
                     strength_score, sentiment_score, confidence, rationale, llm_model, raw_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(speech_id, ticker, llm_model) DO UPDATE SET
                    company_name=excluded.company_name,
                    topic_label=excluded.topic_label,
                    strength=excluded.strength,
                    sentiment=excluded.sentiment,
                    strength_score=excluded.strength_score,
                    sentiment_score=excluded.sentiment_score,
                    confidence=excluded.confidence,
                    rationale=excluded.rationale,
                    raw_response=excluded.raw_response
            ''', (
                speech_id, ticker, ticker_to_name.get(ticker, ticker), topic_label,
                item['strength'], item['sentiment'], strength_score, sentiment_score,
                item['confidence'], item['rationale'], self.model, raw,
            ))
            stored += 1
        return stored


if __name__ == "__main__":
    classifier = GroqTopicClassifier()
    sample = (
        "We must strengthen our digital payments infrastructure and support "
        "banks in expanding financial inclusion across rural India."
    )
    result, _ = classifier.classify_speech(sample, topic_label="Digital & Financial Inclusion",
                                            topic_keywords=["digital", "bank", "payments", "inclusion"])
    print(json.dumps(result, indent=2))
