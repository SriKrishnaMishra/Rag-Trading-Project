import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()


class OptimizedConfig:
    """Centralized configuration management with fallbacks"""

    # API Configuration
    @staticmethod
    def get_groq_key() -> Optional[str]:
        """Get Groq API key with multiple fallback methods"""
        # Method 1: Environment variable
        key = os.getenv('GROQ_API_KEY')
        if key and key != 'your_groq_api_key_here':
            return key

        # Method 2: Streamlit secrets (for deployment)
        try:
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                return st.secrets['GROQ_API_KEY']
        except:
            pass

        return None

    # Market Data Configuration
    DEFAULT_SYMBOLS = [
        # Indian Blue Chips
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        # US Tech Giants
        'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'
    ]

    INDIAN_INDICES = {
        'NIFTY 50': '^NSEI',
        'SENSEX': '^BSESN',
        'NIFTY BANK': '^NSEBANK',
        'NIFTY IT': '^CNXIT',
        'NIFTY AUTO': '^CNXAUTO'
    }

    GLOBAL_INDICES = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOW JONES': '^DJI',
        'FTSE 100': '^FTSE',
        'DAX': '^GDAXI',
        'NIKKEI': '^N225'
    }

    # Technical Analysis Configuration
    TECHNICAL_INDICATORS = {
        'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
        'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
        'MA': {'short': 20, 'medium': 50, 'long': 200},
        'BB': {'period': 20, 'std': 2},
        'STOCH': {'k_period': 14, 'd_period': 3}
    }

    # Performance Configuration
    CACHE_TTL = {
        'market_data': 300,  # 5 minutes
        'news_data': 600,  # 10 minutes
        'symbol_info': 3600,  # 1 hour
        'analysis': 900  # 15 minutes
    }

    # Data Limits
    MAX_SYMBOLS = 10
    MAX_NEWS_ITEMS = 50
    MAX_ANALYSIS_HISTORY = 100

    # News Sources Configuration
    NEWS_SOURCES = [
        {
            'name': 'Yahoo Finance',
            'rss_url': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'weight': 1.0
        },
        {
            'name': 'Reuters Business',
            'rss_url': 'https://feeds.reuters.com/reuters/businessNews',
            'weight': 0.8
        }
    ]

    # AI Configuration
    AI_CONFIG = {
        'model': 'llama-3.1-70b-versatile',
        'max_tokens': 1000,
        'temperature': 0.7,
        'fallback_analysis': True,
        'timeout': 30
    }

    # Styling Configuration
    COLORS = {
        'bull': '#00ff88',
        'bear': '#ff4444',
        'neutral': '#888888',
        'primary': '#667eea',
        'secondary': '#764ba2',
        'warning': '#ffa500',
        'info': '#17a2b8'
    }

    # Error Handling Configuration
    RETRY_CONFIG = {
        'max_retries': 3,
        'backoff_factor': 1.5,
        'timeout': 10
    }

    # Logging Configuration
    LOG_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }

    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """Validate if symbol format is correct"""
        if not symbol or len(symbol.strip()) == 0:
            return False

        symbol = symbol.upper().strip()

        # Basic validation rules
        if len(symbol) > 20:  # Too long
            return False

        # Check for invalid characters
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
        if not all(c in valid_chars for c in symbol):
            return False

        return True

    @classmethod
    def get_symbol_with_exchange(cls, symbol: str) -> str:
        """Auto-add exchange suffix for Indian stocks"""
        symbol = symbol.upper().strip()

        if not symbol:
            return symbol

        # If already has exchange suffix, return as is
        if any(suffix in symbol for suffix in ['.NS', '.BO', '.L', '.TO']):
            return symbol

        # List of common Indian stock symbols that should have .NS
        indian_symbols = {
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'BHARTIARTL',
            'SBIN', 'ITC', 'HINDUNILVR', 'KOTAKBANK', 'LT', 'HCLTECH',
            'ASIANPAINT', 'MARUTI', 'AXISBANK', 'TITAN', 'SUNPHARMA',
            'ULTRACEMCO', 'WIPRO', 'NTPC', 'POWERGRID', 'TATAMOTORS'
        }

        if symbol in indian_symbols:
            return f"{symbol}.NS"

        # For other symbols, return as is (assume US market)
        return symbol

    @classmethod
    def get_display_name(cls, symbol: str) -> str:
        """Get display-friendly name for symbol"""
        # Remove exchange suffix for display
        display_symbol = symbol.replace('.NS', '').replace('.BO', '')
        return display_symbol

    @classmethod
    def get_currency_symbol(cls, symbol: str) -> str:
        """Get currency symbol based on exchange"""
        if '.NS' in symbol or '.BO' in symbol:
            return '₹'
        elif '.L' in symbol:
            return '£'
        elif '.TO' in symbol:
            return 'C$'
        else:
            return '$'

    @classmethod
    def is_market_open(cls, symbol: str) -> bool:
        """Check if market is open (simplified)"""
        from datetime import datetime
        import pytz

        try:
            if '.NS' in symbol or '.BO' in symbol:
                # Indian market: 9:15 AM to 3:30 PM IST, Mon-Fri
                ist = pytz.timezone('Asia/Kolkata')
                now = datetime.now(ist)

                if now.weekday() >= 5:  # Weekend
                    return False

                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

                return market_open <= now <= market_close

            else:
                # US market: 9:30 AM to 4:00 PM EST, Mon-Fri
                est = pytz.timezone('US/Eastern')
                now = datetime.now(est)

                if now.weekday() >= 5:  # Weekend
                    return False

                market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

                return market_open <= now <= market_close

        except:
            # If timezone handling fails, assume market is open
            return True



# import os
# from dotenv import load_dotenv
#
# load_dotenv()
#
#
# class Config:
#     """Configuration management for the trading analysis app"""
#
#     # API Keys (Free services)
#     GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
#     NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')  # newsapi.org free tier
#     ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')  # Free tier available
#
#     # Default settings
#     DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN']
#
#     # Vector DB settings
#     VECTOR_DB_PATH = "./chroma_db"
#     EMBEDDING_MODEL = "llama-3.1-70b-versatile"
#
#     # News sources (free RSS feeds)
#     NEWS_SOURCES = [
#         'https://feeds.finance.yahoo.com/rss/2.0/headline',
#         'https://feeds.reuters.com/reuters/businessNews',
#         'https://rss.cnn.com/rss/money_news_international.rss'
#     ]
#
#     @classmethod
#     def get_groq_key(cls):
#         """Get Groq API key with validation"""
#         key = cls.GROQ_API_KEY
#         if not key:
#             raise ValueError("GROQ_API_KEY not found. Please add it to .env file")
#         return key
#
#     @classmethod
#     def get_news_key(cls):
#         """Get News API key (optional for free sources)"""
#         return cls.NEWS_API_KEY or None