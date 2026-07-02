from .utils.agent_utils import create_msg_delete
from .utils.agent_states import AgentState, InvestDebateState, RiskDebateState

from .analysts.fundamentals_analyst import create_fundamentals_analyst
from .analysts.market_analyst import create_market_analyst
from .analysts.technical_analyst import create_technical_analyst
from .analysts.quant_analyst import create_quant_analyst
from .analysts.sector_analyst import create_sector_analyst
from .analysts.geographic_analyst import create_geographic_analyst
from .analysts.news_analyst import create_news_analyst
from .analysts.sentiment_analyst import (
    create_sentiment_analyst,
    create_social_media_analyst,  # deprecated alias kept for back-compat
)

from .researchers.bear_researcher import create_bear_researcher
from .researchers.bull_researcher import create_bull_researcher

from .risk_mgmt.risk_manager import create_risk_manager
# Legacy debators kept for import back-compat; not wired into the graph.
from .risk_mgmt.aggressive_debator import create_aggressive_debator
from .risk_mgmt.conservative_debator import create_conservative_debator
from .risk_mgmt.neutral_debator import create_neutral_debator

from .managers.research_manager import create_research_manager
from .managers.portfolio_manager import create_portfolio_manager

from .trader.trader import create_trader

__all__ = [
    "AgentState",
    "create_msg_delete",
    "InvestDebateState",
    "RiskDebateState",
    # Analysts
    "create_market_analyst",
    "create_fundamentals_analyst",
    "create_technical_analyst",
    "create_quant_analyst",
    "create_sector_analyst",
    "create_geographic_analyst",
    "create_news_analyst",
    "create_sentiment_analyst",
    "create_social_media_analyst",  # deprecated; will be removed in a future version
    # Researchers
    "create_bear_researcher",
    "create_bull_researcher",
    # Risk management
    "create_risk_manager",
    "create_aggressive_debator",   # legacy; not wired
    "create_conservative_debator", # legacy; not wired
    "create_neutral_debator",      # legacy; not wired
    # Managers
    "create_research_manager",
    "create_portfolio_manager",
    # Trader
    "create_trader",
]
