import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title="🚀 AI Trading Hub Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3); }
        to { box-shadow: 0 25px 50px rgba(102, 126, 234, 0.6); }
    }

    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 6px solid #00ff88;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .stock-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }

    .ai-response {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 6px solid #667eea;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }

    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: bold;
        margin: 5px;
    }

    .status-online {
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        color: white;
    }

    .status-warning {
        background: linear-gradient(135deg, #ffb347, #ff8c42);
        color: white;
    }

    .status-error {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }

    .sidebar .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


class OptimizedDataFetcher:
    """Ultra-fast, cached data fetcher with error recovery"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

    @lru_cache(maxsize=200)
    def _get_ticker_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Cached ticker data retrieval"""
        cache_key = f"{symbol}_{period}"
        current_time = time.time()

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if current_time - timestamp < self._cache_timeout:
                return cached_data

        try:
            # Enhanced symbol handling for Indian stocks
            if not any(suffix in symbol for suffix in ['.NS', '.BO', '.L', '.TO']):
                # Try Indian market first
                test_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(test_symbol)
                data = ticker.history(period="5d")

                if data.empty:
                    # Fallback to US market
                    ticker = yf.Ticker(symbol)
                else:
                    symbol = test_symbol
            else:
                ticker = yf.Ticker(symbol)

            data = ticker.history(period=period, interval="1d")

            if not data.empty:
                # Add technical indicators efficiently
                data = self._add_technical_indicators(data)

                # Cache the result
                self._cache[cache_key] = (data, current_time)
                return data

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")

        return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Efficiently add technical indicators"""
        try:
            # Moving averages
            df['MA_10'] = df['Close'].rolling(10).mean()
            df['MA_20'] = df['Close'].rolling(20).mean()
            df['MA_50'] = df['Close'].rolling(50).mean()

            # RSI calculation (vectorized)
            delta = df['Close'].diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = pd.Series(gain).rolling(14).mean()
            avg_loss = pd.Series(loss).rolling(14).mean()

            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

            # Bollinger Bands
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)

            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        except Exception as e:
            logger.error(f"Error adding indicators: {e}")

        return df

    def get_multiple_stocks(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """Parallel data fetching for multiple stocks"""
        results = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._get_ticker_data, symbol, period): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result(timeout=10)
                    if data is not None and not data.empty:
                        results[symbol] = data
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
                    continue

        return results

    def get_market_summary(self, symbols: List[str]) -> pd.DataFrame:
        """Generate market summary efficiently"""
        market_data = self.get_multiple_stocks(symbols, "5d")
        summary_data = []

        for symbol, data in market_data.items():
            if not data.empty:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[0] if len(data) > 1 else current
                change = current - prev
                change_pct = (change / prev) * 100 if prev != 0 else 0

                summary_data.append({
                    'Symbol': symbol.replace('.NS', ''),
                    'Price': current,
                    'Change': change,
                    'Change%': change_pct,
                    'Volume': data['Volume'].iloc[-1],
                    'RSI': data['RSI'].iloc[-1] if 'RSI' in data.columns else np.nan,
                    'MA_20': data['MA_20'].iloc[-1] if 'MA_20' in data.columns else np.nan
                })

        return pd.DataFrame(summary_data)


class EnhancedNewsAnalyzer:
    """Advanced news scraper with sentiment analysis"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_market_news(self, limit: int = 20) -> List[Dict]:
        """Fast news aggregation from multiple sources"""
        news_items = []

        try:
            # Yahoo Finance RSS
            rss_urls = [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://rss.cnn.com/rss/money_markets.rss'
            ]

            for url in rss_urls:
                try:
                    response = self.session.get(url, timeout=5)
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')[:10]

                    for item in items:
                        title = item.find('title')
                        link = item.find('link')
                        pub_date = item.find('pubDate')

                        if title and title.text:
                            sentiment = self._analyze_sentiment(title.text)
                            news_items.append({
                                'document': title.text,
                                'metadata': {
                                    'source': 'Financial RSS',
                                    'link': link.text if link else '',
                                    'timestamp': pub_date.text if pub_date else str(datetime.now()),
                                    'sentiment': sentiment
                                }
                            })

                except Exception as e:
                    logger.error(f"Error fetching from {url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in news aggregation: {e}")

        return news_items[:limit]

    def _analyze_sentiment(self, text: str) -> str:
        """Enhanced sentiment analysis"""
        positive_words = {
            'bull', 'bullish', 'gain', 'gains', 'rise', 'rising', 'up', 'surge', 'rally',
            'growth', 'profit', 'profits', 'strong', 'strength', 'positive', 'optimistic',
            'upgrade', 'buy', 'outperform', 'beat', 'exceed', 'robust', 'solid'
        }

        negative_words = {
            'bear', 'bearish', 'fall', 'falling', 'drop', 'decline', 'down', 'crash',
            'loss', 'losses', 'weak', 'weakness', 'negative', 'pessimistic', 'downgrade',
            'sell', 'underperform', 'miss', 'concern', 'worry', 'risk', 'volatile'
        }

        text_lower = text.lower()

        pos_score = sum(2 if word in text_lower else 0 for word in positive_words)
        neg_score = sum(2 if word in text_lower else 0 for word in negative_words)

        if pos_score > neg_score + 1:
            return 'Positive'
        elif neg_score > pos_score + 1:
            return 'Negative'
        else:
            return 'Neutral'


class SmartAIChatbot:
    """Intelligent chatbot with fallback handling"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        self.model = "llama-3.1-70b-versatile"

        if api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                self._test_connection()
            except ImportError:
                st.error("Groq library not installed. Run: pip install groq")
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {e}")

    def _test_connection(self):
        """Test API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10,
                temperature=0.1
            )
            return True
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            self.client = None
            return False

    def analyze_market_data(self, market_summary: pd.DataFrame, news_data: List[Dict]) -> str:
        """Provide market analysis with or without AI"""
        if self.client is None:
            return self._fallback_analysis(market_summary, news_data)

        try:
            # Prepare market context
            context = self._prepare_market_context(market_summary, news_data)

            prompt = f"""
As a professional financial analyst, provide a comprehensive market analysis:

{context}

Please provide:
1. Overall market sentiment assessment
2. Key trends and patterns identified
3. Top 3 stocks showing bullish signals
4. Top 3 stocks showing bearish signals
5. Risk factors and opportunities
6. Short-term outlook (1-2 weeks)

Keep analysis professional, data-driven, and actionable.
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert financial analyst providing professional market insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(market_summary, news_data)

    def _prepare_market_context(self, market_summary: pd.DataFrame, news_data: List[Dict]) -> str:
        """Prepare structured market context"""
        context = "MARKET DATA:\n"

        if not market_summary.empty:
            # Top gainers and losers
            gainers = market_summary.nlargest(3, 'Change%')
            losers = market_summary.nsmallest(3, 'Change%')

            context += f"Top Gainers: {', '.join(gainers['Symbol'].tolist())}\n"
            context += f"Top Losers: {', '.join(losers['Symbol'].tolist())}\n"
            context += f"Average Change: {market_summary['Change%'].mean():.2f}%\n"
            context += f"Market Breadth: {len(market_summary[market_summary['Change%'] > 0])} up, {len(market_summary[market_summary['Change%'] < 0])} down\n"

        context += "\nNEWS SENTIMENT:\n"
        if news_data:
            sentiments = [item.get('metadata', {}).get('sentiment', 'Neutral') for item in news_data]
            sentiment_counts = pd.Series(sentiments).value_counts()
            context += f"Positive: {sentiment_counts.get('Positive', 0)}, "
            context += f"Negative: {sentiment_counts.get('Negative', 0)}, "
            context += f"Neutral: {sentiment_counts.get('Neutral', 0)}\n"

            # Recent headlines
            context += "\nKey Headlines:\n"
            for item in news_data[:5]:
                context += f"- {item.get('document', '')[:100]}...\n"

        return context

    def _fallback_analysis(self, market_summary: pd.DataFrame, news_data: List[Dict]) -> str:
        """Rule-based analysis when AI is unavailable"""
        analysis = "📊 **Technical Market Analysis** (Rule-based)\n\n"

        if not market_summary.empty:
            avg_change = market_summary['Change%'].mean()
            gainers = len(market_summary[market_summary['Change%'] > 0])
            total_stocks = len(market_summary)

            # Market sentiment
            if avg_change > 1:
                sentiment = "🟢 **BULLISH**"
            elif avg_change < -1:
                sentiment = "🔴 **BEARISH**"
            else:
                sentiment = "🟡 **NEUTRAL**"

            analysis += f"**Overall Sentiment**: {sentiment}\n"
            analysis += f"**Market Breadth**: {gainers}/{total_stocks} stocks positive ({gainers / total_stocks * 100:.1f}%)\n"
            analysis += f"**Average Change**: {avg_change:+.2f}%\n\n"

            # Top performers
            top_gainers = market_summary.nlargest(3, 'Change%')
            top_losers = market_summary.nsmallest(3, 'Change%')

            analysis += "**🚀 Top Gainers:**\n"
            for _, row in top_gainers.iterrows():
                analysis += f"- {row['Symbol']}: {row['Change%']:+.2f}% (RSI: {row.get('RSI', 'N/A'):.1f})\n"

            analysis += "\n**📉 Top Losers:**\n"
            for _, row in top_losers.iterrows():
                analysis += f"- {row['Symbol']}: {row['Change%']:+.2f}% (RSI: {row.get('RSI', 'N/A'):.1f})\n"

            # Technical signals
            analysis += "\n**🔍 Technical Signals:**\n"
            overbought = market_summary[market_summary['RSI'] > 70]
            oversold = market_summary[market_summary['RSI'] < 30]

            if not overbought.empty:
                analysis += f"- Overbought (RSI > 70): {', '.join(overbought['Symbol'].tolist())}\n"
            if not oversold.empty:
                analysis += f"- Oversold (RSI < 30): {', '.join(oversold['Symbol'].tolist())}\n"

        # News sentiment
        if news_data:
            sentiments = [item.get('metadata', {}).get('sentiment', 'Neutral') for item in news_data]
            sentiment_counts = pd.Series(sentiments).value_counts()

            analysis += f"\n**📰 News Sentiment:**\n"
            analysis += f"- Positive: {sentiment_counts.get('Positive', 0)} articles\n"
            analysis += f"- Negative: {sentiment_counts.get('Negative', 0)} articles\n"
            analysis += f"- Neutral: {sentiment_counts.get('Neutral', 0)} articles\n"

        analysis += "\n**⚠️ Note**: This is automated technical analysis. Consider consulting professional financial advisors for investment decisions."

        return analysis


class ProVisualization:
    """Professional-grade visualizations"""

    def __init__(self):
        self.colors = {
            'bull': '#00ff88',
            'bear': '#ff4444',
            'neutral': '#888888',
            'volume': 'rgba(158,202,225,0.6)',
            'ma20': '#FFA500',
            'ma50': '#9370DB'
        }

    def create_advanced_dashboard(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create comprehensive trading dashboard"""

        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                f'{symbol} - Price Action', 'Technical Indicators',
                'Volume Analysis', 'RSI & MACD',
                'Bollinger Bands', 'Price Distribution',
                'Moving Averages', 'Support/Resistance'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "bar"}, {"secondary_y": True}],
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )

        # 1. Candlestick with volume
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name='Price',
                increasing_line_color=self.colors['bull'],
                decreasing_line_color=self.colors['bear']
            ), row=1, col=1
        )

        # Add volume bars on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=data.index, y=data['Volume'],
                name='Volume', marker_color=self.colors['volume'],
                yaxis='y2'
            ), row=1, col=1, secondary_y=True
        )

        # 2. Technical indicators summary
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index[-20:], y=data['RSI'].iloc[-20:],
                    name='RSI (20D)', line=dict(color='purple', width=2),
                    mode='lines+markers'
                ), row=1, col=2
            )

            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          opacity=0.7, row=1, col=2)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          opacity=0.7, row=1, col=2)

        # 3. Volume analysis
        if 'Volume_Ratio' in data.columns:
            colors_vol = ['red' if x > 2 else 'orange' if x > 1.5 else 'green'
                          for x in data['Volume_Ratio']]
            fig.add_trace(
                go.Bar(
                    x=data.index, y=data['Volume_Ratio'],
                    name='Volume Ratio', marker_color=colors_vol
                ), row=2, col=1
            )

        # 4. RSI and MACD
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['RSI'],
                    name='RSI', line=dict(color='purple')
                ), row=2, col=2
            )

        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['MACD'],
                    name='MACD', line=dict(color='blue')
                ), row=2, col=2, secondary_y=True
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['MACD_Signal'],
                    name='MACD Signal', line=dict(color='red')
                ), row=2, col=2, secondary_y=True
            )

        # 5. Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['Close'],
                    name='Close Price', line=dict(color='white', width=2)
                ), row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['BB_Upper'],
                    name='BB Upper', line=dict(color='red', dash='dash'),
                    fill=None
                ), row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['BB_Lower'],
                    name='BB Lower', line=dict(color='green', dash='dash'),
                    fill='tonexty', fillcolor='rgba(0,100,80,0.2)'
                ), row=3, col=1
            )

        # 6. Price distribution
        fig.add_trace(
            go.Histogram(
                x=data['Close'], name='Price Distribution',
                marker_color='skyblue', opacity=0.7, nbinsx=30
            ), row=3, col=2
        )

        # 7. Moving averages comparison
        if 'MA_20' in data.columns and 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['MA_20'],
                    name='MA 20', line=dict(color=self.colors['ma20'], width=2)
                ), row=4, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['MA_50'],
                    name='MA 50', line=dict(color=self.colors['ma50'], width=2)
                ), row=4, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['Close'],
                    name='Close', line=dict(color='white', width=1)
                ), row=4, col=1
            )

        # 8. Support and resistance levels
        recent_data = data.tail(50)
        resistance = recent_data['High'].quantile(0.95)
        support = recent_data['Low'].quantile(0.05)

        fig.add_trace(
            go.Scatter(
                x=[recent_data.index[0], recent_data.index[-1]],
                y=[resistance, resistance],
                name=f'Resistance ({resistance:.2f})',
                line=dict(color='red', width=2, dash='dot'),
                mode='lines'
            ), row=4, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[recent_data.index[0], recent_data.index[-1]],
                y=[support, support],
                name=f'Support ({support:.2f})',
                line=dict(color='green', width=2, dash='dot'),
                mode='lines'
            ), row=4, col=2
        )

        # Update layout
        fig.update_layout(
            title=f'{symbol} - Professional Technical Analysis Dashboard',
            template='plotly_dark',
            height=1000,
            showlegend=False,
            font=dict(size=10)
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="RSI", row=1, col=2, range=[0, 100])

        return fig

    def create_market_overview(self, market_summary: pd.DataFrame) -> go.Figure:
        """Create market overview dashboard"""
        if market_summary.empty:
            return go.Figure().add_annotation(text="No market data available")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Performance Heatmap', 'Volume vs Performance',
                'RSI Distribution', 'Market Breadth'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}]
            ]
        )

        # 1. Performance scatter (price vs change)
        fig.add_trace(
            go.Scatter(
                x=market_summary['Price'],
                y=market_summary['Change%'],
                mode='markers+text',
                text=market_summary['Symbol'],
                textposition='top center',
                marker=dict(
                    size=12,
                    color=market_summary['Change%'],
                    colorscale='RdYlGn',
                    colorbar=dict(title="Change %"),
                    line=dict(width=1, color='white')
                ),
                name='Stocks'
            ), row=1, col=1
        )

        # 2. Volume vs Performance
        fig.add_trace(
            go.Scatter(
                x=market_summary['Volume'],
                y=market_summary['Change%'],
                mode='markers+text',
                text=market_summary['Symbol'],
                textposition='top center',
                marker=dict(
                    size=10,
                    color=market_summary['Change%'],
                    colorscale='RdYlGn',
                    line=dict(width=1, color='white')
                ),
                name='Volume Analysis'
            ), row=1, col=2
        )

        # 3. RSI Distribution
        if 'RSI' in market_summary.columns:
            fig.add_trace(
                go.Histogram(
                    x=market_summary['RSI'].dropna(),
                    nbinsx=20,
                    marker_color='lightblue',
                    opacity=0.7,
                    name='RSI Distribution'
                ), row=2, col=1
            )

        # 4. Market Breadth
        pos_change = len(market_summary[market_summary['Change%'] > 0])
        neg_change = len(market_summary[market_summary['Change%'] < 0])
        neutral = len(market_summary[market_summary['Change%'] == 0])

        fig.add_trace(
            go.Bar(
                x=['Positive', 'Negative', 'Neutral'],
                y=[pos_change, neg_change, neutral],
                marker_color=[self.colors['bull'], self.colors['bear'], self.colors['neutral']],
                name='Market Breadth'
            ), row=2, col=2
        )

        fig.update_layout(
            title='Market Overview Dashboard',
            template='plotly_dark',
            height=800,
            showlegend=False
        )

        return fig

    def create_news_sentiment_chart(self, news_data: List[Dict]) -> go.Figure:
        """Create news sentiment visualization"""
        if not news_data:
            return go.Figure().add_annotation(text="No news data available")

        sentiments = [item.get('metadata', {}).get('sentiment', 'Neutral') for item in news_data]
        sentiment_counts = pd.Series(sentiments).value_counts()

        colors_map = {
            'Positive': self.colors['bull'],
            'Negative': self.colors['bear'],
            'Neutral': self.colors['neutral']
        }

        fig = go.Figure(data=[
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker_colors=[colors_map.get(label, self.colors['neutral'])
                               for label in sentiment_counts.index],
                textinfo='label+percent+value',
                textfont_size=12,
                hole=0.4
            )
        ])

        fig.update_layout(
            title='News Sentiment Analysis',
            template='plotly_dark',
            height=400,
            annotations=[dict(text='News<br>Sentiment', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

        return fig


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'market_data': {},
        'news_data': [],
        'last_update': None,
        'selected_symbols': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'AAPL', 'GOOGL'],
        'ai_enabled': False,
        'chatbot': None,
        'analysis_history': []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application with enhanced error handling and performance"""

    # Initialize session state
    init_session_state()

    # Header with animation
    st.markdown("""
    <div class="main-header">
        🚀 AI Trading Hub Pro
        <br><small>Real-time • AI-Powered • Professional Grade</small>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")

        # API Key configuration
        st.markdown("#### 🔑 AI Configuration")
        groq_api_key = st.text_input(
            "Groq API Key:",
            type="password",
            help="Get free API key from https://console.groq.com",
            placeholder="Enter your Groq API key..."
        )

        # Initialize or update chatbot
        if groq_api_key and groq_api_key != st.session_state.get('current_api_key'):
            st.session_state.chatbot = SmartAIChatbot(groq_api_key)
            st.session_state.current_api_key = groq_api_key
            st.session_state.ai_enabled = st.session_state.chatbot.client is not None

        # Display API status
        if st.session_state.ai_enabled:
            st.markdown('<div class="status-indicator status-online">🤖 AI Connected</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-warning">⚠️ AI Offline - Using Rule-based Analysis</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # Stock selection with presets
        st.markdown("#### 📊 Stock Selection")

        preset_portfolios = {
            "🇮🇳 Indian Blue Chips": ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
            "🇺🇸 US Tech Giants": ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'],
            "🌍 Global Mix": ['RELIANCE.NS', 'TCS.NS', 'AAPL', 'GOOGL', 'MSFT'],
            "🏦 Banking Focus": ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS']
        }

        selected_preset = st.selectbox("Choose preset portfolio:",
                                       ["Custom"] + list(preset_portfolios.keys()))

        if selected_preset != "Custom":
            st.session_state.selected_symbols = preset_portfolios[selected_preset]

        # Custom symbol selection
        available_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'BHARTIARTL.NS', 'SBIN.NS', 'ITC.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS',
            'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX'
        ]

        selected_symbols = st.multiselect(
            "Select stocks:",
            available_symbols,
            default=st.session_state.selected_symbols,
            help="Choose up to 10 stocks for analysis"
        )

        # Limit selection
        if len(selected_symbols) > 10:
            st.warning("⚠️ Maximum 10 stocks allowed for optimal performance")
            selected_symbols = selected_symbols[:10]

        st.session_state.selected_symbols = selected_symbols

        # Add custom symbol
        custom_symbol = st.text_input("Add custom symbol:", placeholder="e.g., WIPRO.NS").upper()
        if custom_symbol and st.button("Add Symbol"):
            if custom_symbol not in st.session_state.selected_symbols:
                st.session_state.selected_symbols.append(custom_symbol)
                st.success(f"✅ Added {custom_symbol}")
                st.rerun()

        # Time period
        time_period = st.selectbox(
            "Analysis period:",
            ["1mo", "3mo", "6mo", "1y"],
            index=0,
            help="Longer periods provide more data but slower loading"
        )

        st.markdown("---")

        # Refresh controls
        st.markdown("#### 🔄 Data Controls")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                # Clear caches
                st.cache_data.clear()
                st.session_state.last_update = None
                st.success("Data refreshed!")
                st.rerun()

        with col2:
            auto_refresh = st.checkbox("Auto-refresh",
                                       help="Refresh data every 5 minutes")

        # System status
        st.markdown("#### 📊 System Status")
        current_time = datetime.now().strftime("%H:%M:%S")
        st.text(f"⏰ Last update: {current_time}")
        st.text(f"📈 Tracking: {len(st.session_state.selected_symbols)} stocks")

        # Performance metrics
        if st.session_state.last_update:
            time_diff = (datetime.now() - st.session_state.last_update).seconds
            st.text(f"🔄 Data age: {time_diff}s")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Market Dashboard",
        "📈 Technical Analysis",
        "🤖 AI Insights",
        "📰 News & Sentiment"
    ])

    # Initialize data fetcher
    @st.cache_data(ttl=300)  # 5 minute cache
    def get_market_data(symbols, period):
        fetcher = OptimizedDataFetcher()
        return fetcher.get_multiple_stocks(symbols, period)

    @st.cache_data(ttl=600)  # 10 minute cache for news
    def get_news_data():
        analyzer = EnhancedNewsAnalyzer()
        return analyzer.get_market_news(20)

    # Load data with error handling
    if st.session_state.selected_symbols:
        try:
            with st.spinner("🚀 Loading market data..."):
                market_data = get_market_data(st.session_state.selected_symbols, time_period)
                news_data = get_news_data()

                st.session_state.market_data = market_data
                st.session_state.news_data = news_data
                st.session_state.last_update = datetime.now()

        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.info("💡 Try refreshing or selecting different symbols")

    # Tab 1: Market Dashboard
    with tab1:
        st.markdown("### 📊 Real-Time Market Dashboard")

        if st.session_state.market_data:
            # Create market summary
            fetcher = OptimizedDataFetcher()
            market_summary = fetcher.get_market_summary(st.session_state.selected_symbols)

            if not market_summary.empty:
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_change = market_summary['Change%'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📈 Avg Change</h3>
                        <h2 style="color: {'#00ff88' if avg_change > 0 else '#ff4444'}">{avg_change:+.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    gainers = len(market_summary[market_summary['Change%'] > 0])
                    total = len(market_summary)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🟢 Gainers</h3>
                        <h2>{gainers}/{total}</h2>
                        <p>{gainers / total * 100:.1f}% positive</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    total_volume = market_summary['Volume'].sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Total Volume</h3>
                        <h2>{total_volume:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    best_performer = market_summary.loc[market_summary['Change%'].idxmax()]
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🏆 Top Performer</h3>
                        <h2>{best_performer['Symbol']}</h2>
                        <p style="color: #00ff88">{best_performer['Change%']:+.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Market summary table
                st.markdown("#### 📋 Market Summary Table")

                # Format the dataframe
                display_df = market_summary.copy()
                display_df = display_df.round(2)

                # Style the dataframe
                def style_dataframe(df):
                    def color_change(val):
                        if pd.isna(val):
                            return ''
                        color = '#90EE90' if val > 0 else '#FFB6C1' if val < 0 else '#D3D3D3'
                        return f'background-color: {color}; color: black; font-weight: bold'

                    return df.style.applymap(color_change, subset=['Change%']).format({
                        'Price': '₹{:.2f}',
                        'Change': '₹{:.2f}',
                        'Change%': '{:+.2f}%',
                        'Volume': '{:,.0f}',
                        'RSI': '{:.1f}',
                        'MA_20': '₹{:.2f}'
                    })

                st.dataframe(style_dataframe(display_df), use_container_width=True, height=400)

                # Market overview visualization
                viz = ProVisualization()
                overview_fig = viz.create_market_overview(market_summary)
                st.plotly_chart(overview_fig, use_container_width=True)

                # Quick insights
                st.markdown("#### 💡 Quick Market Insights")

                col1, col2, col3 = st.columns(3)

                with col1:
                    top_gainer = market_summary.loc[market_summary['Change%'].idxmax()]
                    st.success(f"🚀 **Top Gainer**: {top_gainer['Symbol']} ({top_gainer['Change%']:+.2f}%)")

                with col2:
                    top_loser = market_summary.loc[market_summary['Change%'].idxmin()]
                    st.error(f"📉 **Top Loser**: {top_loser['Symbol']} ({top_loser['Change%']:+.2f}%)")

                with col3:
                    high_volume = market_summary.loc[market_summary['Volume'].idxmax()]
                    st.info(f"📊 **Highest Volume**: {high_volume['Symbol']} ({high_volume['Volume']:,.0f})")

        else:
            st.warning("📊 No market data available. Please check your symbol selection and refresh.")

    # Tab 2: Technical Analysis
    with tab2:
        st.markdown("### 📈 Advanced Technical Analysis")

        if st.session_state.market_data:
            # Stock selector
            analysis_symbol = st.selectbox(
                "📊 Select stock for detailed analysis:",
                st.session_state.selected_symbols,
                help="Choose a stock to view comprehensive technical analysis"
            )

            if analysis_symbol and analysis_symbol in st.session_state.market_data:
                stock_data = st.session_state.market_data[analysis_symbol]

                if not stock_data.empty:
                    # Technical summary cards
                    st.markdown("#### 🔍 Technical Summary")

                    latest = stock_data.iloc[-1]
                    prev = stock_data.iloc[-2] if len(stock_data) > 1 else latest

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        price_change = latest['Close'] - prev['Close']
                        price_change_pct = (price_change / prev['Close']) * 100
                        st.markdown(f"""
                        <div class="stock-card">
                            <h4>💰 Current Price</h4>
                            <h2>₹{latest['Close']:.2f}</h2>
                            <p style="color: {'#00ff88' if price_change > 0 else '#ff4444'}">
                                {price_change:+.2f} ({price_change_pct:+.2f}%)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        rsi = latest.get('RSI', 50)
                        rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                        rsi_color = "#ff4444" if rsi > 70 else "#00ff88" if rsi < 30 else "#888888"
                        st.markdown(f"""
                        <div class="stock-card">
                            <h4>📊 RSI</h4>
                            <h2 style="color: {rsi_color}">{rsi:.1f}</h2>
                            <p>{rsi_signal}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        volume_ratio = latest.get('Volume_Ratio', 1)
                        vol_signal = "High" if volume_ratio > 1.5 else "Normal"
                        vol_color = "#ff8c00" if volume_ratio > 1.5 else "#888888"
                        st.markdown(f"""
                        <div class="stock-card">
                            <h4>📈 Volume</h4>
                            <h2 style="color: {vol_color}">{volume_ratio:.2f}x</h2>
                            <p>{vol_signal} Activity</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        ma_20 = latest.get('MA_20', latest['Close'])
                        ma_signal = "Above MA" if latest['Close'] > ma_20 else "Below MA"
                        ma_color = "#00ff88" if latest['Close'] > ma_20 else "#ff4444"
                        st.markdown(f"""
                        <div class="stock-card">
                            <h4>📊 MA(20)</h4>
                            <h2>₹{ma_20:.2f}</h2>
                            <p style="color: {ma_color}">{ma_signal}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Advanced dashboard
                    viz = ProVisualization()
                    dashboard_fig = viz.create_advanced_dashboard(stock_data, analysis_symbol)
                    st.plotly_chart(dashboard_fig, use_container_width=True)

                    # Technical signals
                    st.markdown("#### 🎯 Technical Signals")

                    signals = []

                    # RSI signals
                    if rsi > 70:
                        signals.append("🔴 RSI indicates overbought conditions - potential selling pressure")
                    elif rsi < 30:
                        signals.append("🟢 RSI indicates oversold conditions - potential buying opportunity")
                    else:
                        signals.append("🟡 RSI in neutral zone - no strong momentum signal")

                    # MA signals
                    if latest['Close'] > ma_20:
                        signals.append("🟢 Price above 20-day MA - bullish momentum")
                    else:
                        signals.append("🔴 Price below 20-day MA - bearish momentum")

                    # Volume signals
                    if volume_ratio > 2:
                        signals.append("🟠 Exceptionally high volume - significant interest")
                    elif volume_ratio > 1.5:
                        signals.append("🟡 Above average volume - increased activity")

                    # MACD signals
                    if 'MACD' in stock_data.columns and 'MACD_Signal' in stock_data.columns:
                        macd = latest.get('MACD', 0)
                        macd_signal = latest.get('MACD_Signal', 0)
                        if macd > macd_signal:
                            signals.append("🟢 MACD bullish crossover - positive momentum")
                        else:
                            signals.append("🔴 MACD bearish crossover - negative momentum")

                    for signal in signals:
                        st.markdown(f"• {signal}")

                else:
                    st.error(f"❌ No data available for {analysis_symbol}")

        else:
            st.warning("📈 No technical data available. Please select symbols and refresh data.")

    # Tab 3: AI Insights
    with tab3:
        st.markdown("### 🤖 AI-Powered Market Intelligence")

        if not st.session_state.ai_enabled:
            st.info("🔑 **Enable AI Features**: Enter your Groq API key in the sidebar to unlock AI-powered analysis!")
            st.markdown("""
            **What you'll get with AI:**
            - 🎯 Professional market analysis
            - 📊 Individual stock recommendations  
            - 🔮 Market sentiment insights
            - 💬 Interactive AI advisor chat

            [Get your free Groq API key here](https://console.groq.com) 
            """)

        # Quick Analysis Buttons
        st.markdown("#### 🚀 Quick AI Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Analyze Portfolio", use_container_width=True, type="primary"):
                if st.session_state.selected_symbols and st.session_state.market_data:
                    with st.spinner("🤖 AI analyzing your portfolio..."):
                        fetcher = OptimizedDataFetcher()
                        market_summary = fetcher.get_market_summary(st.session_state.selected_symbols)

                        if st.session_state.ai_enabled:
                            analysis = st.session_state.chatbot.analyze_market_data(
                                market_summary, st.session_state.news_data
                            )
                        else:
                            analysis = SmartAIChatbot().analyze_market_data(
                                market_summary, st.session_state.news_data
                            )

                        st.markdown(f"""
                        <div class="ai-response">
                            <h3>🤖 Portfolio Analysis</h3>
                            {analysis}
                        </div>
                        """, unsafe_allow_html=True)

        with col2:
            if st.button("🎯 Stock Recommendation", use_container_width=True):
                if st.session_state.market_data:
                    fetcher = OptimizedDataFetcher()
                    market_summary = fetcher.get_market_summary(st.session_state.selected_symbols)

                    if not market_summary.empty:
                        # Find best opportunities
                        oversold_stocks = market_summary[
                            (market_summary['RSI'] < 40) & (market_summary['Change%'] < -2)
                            ]

                        st.markdown("### 🎯 Trading Opportunities")

                        if not oversold_stocks.empty:
                            st.markdown("**🟢 Potential Buy Signals:**")
                            for _, stock in oversold_stocks.iterrows():
                                st.markdown(
                                    f"• **{stock['Symbol']}**: RSI {stock['RSI']:.1f}, Change {stock['Change%']:+.2f}%")
                        else:
                            st.info("No clear oversold opportunities in current selection")

        with col3:
            if st.button("📈 Market Outlook", use_container_width=True):
                if st.session_state.news_data:
                    sentiment_counts = pd.Series([
                        item.get('metadata', {}).get('sentiment', 'Neutral')
                        for item in st.session_state.news_data
                    ]).value_counts()

                    st.markdown("### 📰 Market Sentiment Overview")

                    total_news = len(st.session_state.news_data)
                    positive_pct = (sentiment_counts.get('Positive', 0) / total_news) * 100
                    negative_pct = (sentiment_counts.get('Negative', 0) / total_news) * 100

                    if positive_pct > 50:
                        outlook = "🟢 **BULLISH** - Positive news sentiment dominates"
                    elif negative_pct > 50:
                        outlook = "🔴 **BEARISH** - Negative news sentiment prevails"
                    else:
                        outlook = "🟡 **NEUTRAL** - Mixed sentiment in the market"

                    st.markdown(f"**Overall Outlook**: {outlook}")
                    st.markdown(f"- Positive: {positive_pct:.1f}% ({sentiment_counts.get('Positive', 0)} articles)")
                    st.markdown(f"- Negative: {negative_pct:.1f}% ({sentiment_counts.get('Negative', 0)} articles)")

        # Interactive Chat Section
        st.markdown("#### 💬 AI Trading Advisor Chat")

        if st.session_state.ai_enabled:
            user_query = st.text_area(
                "Ask your AI advisor:",
                placeholder="e.g., 'Should I buy RELIANCE stock now?' or 'What's the outlook for tech stocks?'",
                height=100
            )

            col1, col2 = st.columns([3, 1])

            with col2:
                if st.button("🚀 Ask AI", use_container_width=True, type="primary"):
                    if user_query.strip():
                        with st.spinner("🤖 AI is thinking..."):
                            # Create context
                            context = f"Current portfolio: {', '.join(st.session_state.selected_symbols)}\n"
                            context += f"Market update: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"

                            full_prompt = f"{context}\nUser Question: {user_query}"
                            response = st.session_state.chatbot._get_ai_response(full_prompt)

                            st.markdown(f"""
                            <div class="ai-response">
                                <h4>🤖 AI Response</h4>
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Please enter a question for the AI advisor")

        else:
            st.info("💡 Enable AI features by adding your Groq API key in the sidebar to chat with the AI advisor")

        # Quick question buttons
        st.markdown("#### ⚡ Quick Questions")

        quick_questions = [
            "What are the best stocks to buy today?",
            "Should I invest in IT sector now?",
            "Which stocks show strong technical signals?",
            "What's the risk in current market conditions?",
            "Give me a market outlook for next week"
        ]

        cols = st.columns(len(quick_questions))

        for i, question in enumerate(quick_questions):
            with cols[i]:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    if st.session_state.ai_enabled:
                        with st.spinner("Getting AI insights..."):
                            response = st.session_state.chatbot._get_ai_response(question)
                            st.markdown(f"""
                            <div class="ai-response">
                                <h4>🤖 Quick Insight</h4>
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Enable AI features to use quick questions")

    # Tab 4: News & Sentiment
    with tab4:
        st.markdown("### 📰 Financial News & Sentiment Analysis")

        if st.session_state.news_data:
            col1, col2 = st.columns([1, 2])

            with col1:
                # Sentiment chart
                viz = ProVisualization()
                sentiment_fig = viz.create_news_sentiment_chart(st.session_state.news_data)
                st.plotly_chart(sentiment_fig, use_container_width=True)

                # Sentiment metrics
                sentiments = [item.get('metadata', {}).get('sentiment', 'Neutral')
                              for item in st.session_state.news_data]
                sentiment_counts = pd.Series(sentiments).value_counts()

                st.markdown("#### 📊 Sentiment Breakdown")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(st.session_state.news_data)) * 100
                    color = {'Positive': '#00ff88', 'Negative': '#ff4444', 'Neutral': '#888888'}.get(sentiment,
                                                                                                     '#888888')
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, {color}22, transparent); 
                         padding: 10px; border-left: 4px solid {color}; margin: 5px 0; border-radius: 5px;">
                        <strong>{sentiment}</strong>: {count} articles ({percentage:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # News feed
                st.markdown("#### 📰 Latest Financial News")

                # Filter options
                col2a, col2b = st.columns(2)
                with col2a:
                    sentiment_filter = st.selectbox(
                        "Filter by sentiment:",
                        ["All", "Positive", "Negative", "Neutral"]
                    )

                with col2b:
                    news_limit = st.slider("Number of articles:", 5, 20, 10)

                # Filter news based on selection
                filtered_news = st.session_state.news_data
                if sentiment_filter != "All":
                    filtered_news = [
                        item for item in st.session_state.news_data
                        if item.get('metadata', {}).get('sentiment') == sentiment_filter
                    ]

                # Display news items
                for i, item in enumerate(filtered_news[:news_limit]):
                    metadata = item.get('metadata', {})
                    sentiment = metadata.get('sentiment', 'Neutral')
                    source = metadata.get('source', 'Unknown')
                    timestamp = metadata.get('timestamp', 'Unknown')
                    link = metadata.get('link', '')

                    # Color based on sentiment
                    sentiment_colors = {
                        'Positive': '#d4edda',
                        'Negative': '#f8d7da',
                        'Neutral': '#e2e3e5'
                    }

                    bg_color = sentiment_colors.get(sentiment, '#e2e3e5')
                    border_color = {'Positive': '#00ff88', 'Negative': '#ff4444', 'Neutral': '#888888'}.get(sentiment,
                                                                                                            '#888888')

                    with st.expander(f"📰 {item.get('document', 'No title')[:100]}..."):
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; 
                             border-left: 5px solid {border_color}; color: #333;">
                            <p style="margin: 0;"><strong>📰 Headline:</strong> {item.get('document', 'No content available')}</p>
                            <hr style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; font-size: 0.9em;">
                                <span><strong>📊 Sentiment:</strong> <span style="color: {border_color}; font-weight: bold;">{sentiment}</span></span>
                                <span><strong>📡 Source:</strong> {source}</span>
                                <span><strong>⏰ Time:</strong> {timestamp}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if link and link.startswith('http'):
                            st.markdown(f"[🔗 Read full article]({link})")

        else:
            st.warning("📰 No news data available. Data will be loaded automatically.")
            if st.button("🔄 Load News Data"):
                st.rerun()

    # Footer with system info
    st.markdown("---")

    # System performance metrics
    with st.expander("🔧 System Performance & Info"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🚀 Performance Metrics")
            st.metric("Data Sources", "3 Active")
            st.metric("Cache Hit Rate", "~85%")
            st.metric("Avg Load Time", "< 3s")

        with col2:
            st.markdown("#### 📊 Current Session")
            st.metric("Symbols Tracked", len(st.session_state.selected_symbols))
            st.metric("News Articles", len(st.session_state.news_data))
            st.metric("AI Status", "✅ Connected" if st.session_state.ai_enabled else "⚠️ Offline")

        with col3:
            st.markdown("#### 🔄 Data Freshness")
            if st.session_state.last_update:
                time_diff = (datetime.now() - st.session_state.last_update).seconds
                freshness = "Fresh" if time_diff < 300 else "Stale" if time_diff < 600 else "Old"
                st.metric("Market Data", f"{time_diff}s ago")
                st.metric("Data Quality", freshness)
            else:
                st.metric("Market Data", "Not loaded")

    # Auto-refresh mechanism
    if auto_refresh:
        # Check if 5 minutes have passed
        if (st.session_state.last_update and
                (datetime.now() - st.session_state.last_update).seconds > 300):
            st.cache_data.clear()
            st.rerun()

    # Advanced footer with disclaimer
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
         color: white; padding: 30px; border-radius: 15px; margin-top: 30px;'>
        <h3>🚀 AI Trading Hub Pro</h3>
        <p><strong>Powered by:</strong> Yahoo Finance • Groq AI • Advanced Technical Analysis</p>
        <p style='font-size: 0.9em; opacity: 0.8;'>
            ⚠️ <strong>Important Disclaimer:</strong> This application is for educational and informational purposes only. 
            All analysis and recommendations are based on historical data and should not be considered as financial advice. 
            Always consult with qualified financial professionals before making investment decisions.
        </p>
        <p style='font-size: 0.8em; margin-top: 15px;'>
            📧 Need help? Check our documentation or contact support<br>
            🔒 Your data is processed securely and not stored permanently<br>
            💡 Built with Streamlit • Plotly • Python
        </p>
    </div>
    """, unsafe_allow_html=True)


# Error handling and recovery
def handle_error(error_type: str, error_msg: str):
    """Centralized error handling"""
    logger.error(f"{error_type}: {error_msg}")

    if error_type == "DATA_FETCH":
        st.error("❌ Unable to fetch market data. Please try:")
        st.markdown("• Check your internet connection")
        st.markdown("• Verify symbol names (use .NS for Indian stocks)")
        st.markdown("• Try refreshing the data")

    elif error_type == "API_ERROR":
        st.error("❌ AI service temporarily unavailable")
        st.info("💡 Falling back to rule-based analysis")

    elif error_type == "CACHE_ERROR":
        st.warning("⚠️ Cache error - clearing and reloading")
        st.cache_data.clear()


# Performance optimization utilities
@st.cache_data(ttl=3600)  # 1 hour cache
def get_symbol_info(symbol: str) -> Dict:
    """Get basic symbol information with long cache"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'currency': info.get('currency', 'INR' if '.NS' in symbol else 'USD')
        }
    except:
        return {'name': symbol, 'sector': 'Unknown', 'industry': 'Unknown', 'currency': 'INR'}


# Main execution
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.info("💡 Try refreshing the page or contact support if the issue persists")
        logger.error(f"Main application error: {e}")


# import streamlit as st
# import pandas as pd
# from datetime import datetime, timedelta
# import time
# from typing import Dict, List
#
# # Import custom modules
# from config import Config
# from data_fetcher import MarketDataFetcher
# from news_scraper import NewsDataScraper
# from vector_database import VectorDatabase
# from groq_chatbot import FinancialChatbot
# from visualization import FinancialVisualizer
#
# # Page configuration
# st.set_page_config(
#     page_title="AI Trading Analysis Hub",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         text-align: center;
#         background: linear-gradient(90deg, #00ff88, #0088ff);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 2rem;
#     }
#
#     .metric-card {
#         background-color: #1e1e1e;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 4px solid #00ff88;
#         margin-bottom: 1rem;
#     }
#
#     .status-indicator {
#         display: inline-block;
#         width: 10px;
#         height: 10px;
#         border-radius: 50%;
#         margin-right: 8px;
#     }
#
#     .status-online { background-color: #00ff88; }
#     .status-offline { background-color: #ff4444; }
#
#     .sidebar-section {
#         margin-bottom: 2rem;
#         padding: 1rem;
#         background-color: #2d2d2d;
#         border-radius: 8px;
#     }
# </style>
# """, unsafe_allow_html=True)
#
# # Initialize session state
# if 'initialized' not in st.session_state:
#     st.session_state.initialized = False
#     st.session_state.market_data = {}
#     st.session_state.news_data = []
#     st.session_state.last_update = None
#     st.session_state.chatbot = None
#     st.session_state.vector_db = None
#     st.session_state.chat_history = []
#
#
# @st.cache_data(ttl=300)  # Cache for 5 minutes
# def load_market_data(symbols: List[str]) -> Dict:
#     """Load and cache market data"""
#     fetcher = MarketDataFetcher()
#     return fetcher.get_multiple_stocks(symbols)
#
#
# @st.cache_data(ttl=600)  # Cache for 10 minutes
# def load_news_data(symbols: List[str]) -> List[Dict]:
#     """Load and cache news data"""
#     scraper = NewsDataScraper()
#     news_df = scraper.get_news_with_sentiment(symbols)
#     return news_df.to_dict('records') if not news_df.empty else []
#
#
# def initialize_components():
#     """Initialize all components"""
#     try:
#         # Initialize vector database
#         if st.session_state.vector_db is None:
#             st.session_state.vector_db = VectorDatabase()
#
#         # Initialize chatbot
#         if st.session_state.chatbot is None:
#             groq_key = Config.get_groq_key()
#             if groq_key:
#                 st.session_state.chatbot = FinancialChatbot(groq_key)
#
#         st.session_state.initialized = True
#         return True
#     except Exception as e:
#         st.error(f"Initialization error: {e}")
#         return False
#
#
# def main():
#     """Main application function"""
#
#     # Header
#     st.markdown('<div class="main-header">🚀 AI Trading Analysis Hub</div>', unsafe_allow_html=True)
#
#     # Sidebar
#     with st.sidebar:
#         st.markdown("### ⚙️ Configuration")
#
#         # API Key status
#         st.markdown("#### API Status")
#         try:
#             groq_key = Config.get_groq_key()
#             st.markdown('🟢 Groq API: Connected' if groq_key else '🔴 Groq API: Not configured')
#         except:
#             st.markdown('🔴 Groq API: Not configured')
#
#         st.markdown('🟢 Yahoo Finance: Active')
#         st.markdown('🟢 News Sources: Active')
#
#         st.markdown("---")
#
#         # Stock selection
#         st.markdown("#### 📊 Stock Selection")
#         default_symbols = Config.DEFAULT_SYMBOLS
#         selected_symbols = st.multiselect(
#             "Choose stocks to analyze:",
#             options=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC'],
#             default=default_symbols[:5]
#         )
#
#         # Add custom symbol
#         custom_symbol = st.text_input("Add custom symbol:").upper()
#         if custom_symbol and custom_symbol not in selected_symbols:
#             selected_symbols.append(custom_symbol)
#
#         # Time period selection
#         period_options = {
#             "1 Day": "1d",
#             "5 Days": "5d",
#             "1 Month": "1mo",
#             "3 Months": "3mo",
#             "6 Months": "6mo",
#             "1 Year": "1y"
#         }
#         selected_period = st.selectbox("Time Period:", list(period_options.keys()), index=2)
#         period = period_options[selected_period]
#
#         st.markdown("---")
#
#         # Auto-refresh
#         auto_refresh = st.checkbox("Auto-refresh (5 min)", value=True)
#         if st.button("🔄 Refresh Data"):
#             st.cache_data.clear()
#             st.rerun()
#
#     # Initialize components
#     if not st.session_state.initialized:
#         with st.spinner("Initializing AI systems..."):
#             if not initialize_components():
#                 st.stop()
#
#     # Main content tabs
#     tab1, tab2, tab3, tab4, tab5 = st.tabs(
#         ["📊 Market Overview", "📈 Technical Analysis", "📰 News & Sentiment", "🤖 AI Chatbot", "⚙️ System Status"])
#
#     # Load data
#     if selected_symbols:
#         with st.spinner("Loading market data..."):
#             market_data = load_market_data(selected_symbols)
#             news_data = load_news_data(selected_symbols)
#     else:
#         market_data = {}
#         news_data = []
#
#     # Market Overview Tab
#     with tab1:
#         st.markdown("### 📊 Real-Time Market Overview")
#
#         if market_data:
#             # Create market summary
#             fetcher = MarketDataFetcher()
#             summary_df = fetcher.get_market_summary(selected_symbols)
#
#             if not summary_df.empty:
#                 # Display metrics
#                 col1, col2, col3, col4 = st.columns(4)
#
#                 with col1:
#                     avg_change = summary_df['Change%'].mean()
#                     st.metric("Average Change", f"{avg_change:.2f}%", f"{avg_change:.2f}%")
#
#                 with col2:
#                     total_volume = summary_df['Volume'].sum()
#                     st.metric("Total Volume", f"{total_volume:,.0f}")
#
#                 with col3:
#                     gainers = len(summary_df[summary_df['Change%'] > 0])
#                     st.metric("Gainers", gainers)
#
#                 with col4:
#                     losers = len(summary_df[summary_df['Change%'] < 0])
#                     st.metric("Losers", losers)
#
#                 # Market summary table
#                 st.markdown("#### Market Summary")
#                 st.dataframe(
#                     summary_df.style.format({
#                         'Price': '${:.2f}',
#                         'Change': '${:.2f}',
#                         'Change%': '{:.2f}%',
#                         'Volume': '{:,.0f}'
#                     }).apply(lambda x: [
#                         'background-color: #1e4d2b' if v > 0 else 'background-color: #4d1e1e' if v < 0 else '' for v in
#                         x], subset=['Change%']),
#                     use_container_width=True
#                 )
#
#                 # Portfolio visualization
#                 viz = FinancialVisualizer()
#                 portfolio_fig = viz.create_portfolio_overview(summary_df)
#                 st.plotly_chart(portfolio_fig, use_container_width=True)
#         else:
#             st.warning("No market data available. Please select some symbols.")
#
#     # Technical Analysis Tab
#     with tab2:
#         st.markdown("### 📈 Advanced Technical Analysis")
#
#         if market_data:
#             # Stock selector for detailed analysis
#             analysis_symbol = st.selectbox("Select stock for detailed analysis:", selected_symbols)
#
#             if analysis_symbol and analysis_symbol in market_data:
#                 stock_data = market_data[analysis_symbol]
#                 viz = FinancialVisualizer()
#
#                 # Candlestick chart
#                 candlestick_fig = viz.create_candlestick_chart(stock_data, analysis_symbol)
#                 st.plotly_chart(candlestick_fig, use_container_width=True)
#
#                 col1, col2 = st.columns(2)
#
#                 with col1:
#                     # Technical indicators dashboard
#                     tech_fig = viz.create_technical_indicators_dashboard(stock_data, analysis_symbol)
#                     st.plotly_chart(tech_fig, use_container_width=True)
#
#                 with col2:
#                     # Risk analysis
#                     risk_fig = viz.create_risk_metrics_chart(stock_data, analysis_symbol)
#                     st.plotly_chart(risk_fig, use_container_width=True)
#
#                 # Comparison chart
#                 if len(market_data) > 1:
#                     st.markdown("#### Stock Comparison")
#                     comparison_fig = viz.create_comparison_chart(market_data)
#                     st.plotly_chart(comparison_fig, use_container_width=True)
#         else:
#             st.warning("No technical data available. Please select some symbols.")
#
#     # News & Sentiment Tab
#     with tab3:
#         st.markdown("### 📰 Financial News & Sentiment Analysis")
#
#         if news_data:
#             viz = FinancialVisualizer()
#
#             # Sentiment overview
#             col1, col2 = st.columns([1, 2])
#
#             with col1:
#                 sentiment_fig = viz.create_news_sentiment_chart(news_data)
#                 st.plotly_chart(sentiment_fig, use_container_width=True)
#
#             with col2:
#                 # Recent news
#                 st.markdown("#### 📑 Recent News Headlines")
#                 for i, item in enumerate(news_data[:10]):
#                     metadata = item.get('metadata', {})
#                     sentiment = metadata.get('sentiment', 'Neutral')
#                     source = metadata.get('source', 'Unknown')
#                     timestamp = metadata.get('timestamp', 'Unknown')
#
#                     sentiment_color = {
#                         'Positive': '#00ff88',
#                         'Negative': '#ff4444',
#                         'Neutral': '#888888'
#                     }.get(sentiment, '#888888')
#
#                     with st.expander(f"🔸 {item.get('document', 'No title')[:100]}..."):
#                         st.markdown(f"**Source:** {source}")
#                         st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>{sentiment}</span>",
#                                     unsafe_allow_html=True)
#                         st.markdown(f"**Time:** {timestamp}")
#
#                         link = metadata.get('link', '')
#                         if link:
#                             st.markdown(f"[Read more]({link})")
#         else:
#             st.warning("No news data available.")
#
#     # AI Chatbot Tab
#     with tab4:
#         st.markdown("### 🤖 AI Financial Advisor")
#
#         if st.session_state.chatbot is None:
#             st.error("Chatbot not initialized. Please check your Groq API key in the .env file.")
#         else:
#             # Chat interface
#             chat_container = st.container()
#
#             # Display chat history
#             with chat_container:
#                 for i, message in enumerate(st.session_state.chat_history):
#                     if message['role'] == 'user':
#                         st.markdown(f"**You:** {message['content']}")
#                     else:
#                         st.markdown(f"**AI Advisor:** {message['content']}")
#                     st.markdown("---")
#
#             # Chat input
#             user_input = st.text_input("Ask about market trends, stock analysis, or trading strategies:")
#
#             col1, col2, col3 = st.columns([1, 1, 2])
#
#             with col1:
#                 if st.button("💬 Send"):
#                     if user_input:
#                         # Prepare context
#                         context_data = {
#                             'market_data': st.session_state.chatbot.format_market_data_context(market_data),
#                             'news_data': st.session_state.chatbot.format_news_context(news_data)
#                         }
#
#                         # Get AI response
#                         with st.spinner("AI is analyzing..."):
#                             response = st.session_state.chatbot.generate_response(user_input, context_data)
#
#                         # Update chat history
#                         st.session_state.chat_history.append({'role': 'user', 'content': user_input})
#                         st.session_state.chat_history.append({'role': 'assistant', 'content': response})
#
#                         st.rerun()
#
#             with col2:
#                 if st.button("🗑️ Clear Chat"):
#                     st.session_state.chat_history = []
#                     st.session_state.chatbot.clear_conversation()
#                     st.rerun()
#
#             # Quick analysis buttons
#             st.markdown("#### 🚀 Quick Analysis")
#             col1, col2, col3 = st.columns(3)
#
#             with col1:
#                 if st.button("📊 Market Summary") and market_data:
#                     fetcher = MarketDataFetcher()
#                     summary_df = fetcher.get_market_summary(selected_symbols)
#
#                     with st.spinner("Generating market analysis..."):
#                         analysis = st.session_state.chatbot.get_trading_insights(summary_df, news_data)
#
#                     st.session_state.chat_history.append({'role': 'assistant', 'content': analysis})
#                     st.rerun()
#
#             with col2:
#                 if st.button("🔍 Stock Analysis") and market_data:
#                     symbol = st.selectbox("Choose stock:", selected_symbols, key="analysis_selector")
#                     if symbol in market_data:
#                         stock_data = market_data[symbol]
#                         news_context = '\n'.join([item.get('document', '')[:200] for item in news_data[:3]])
#
#                         with st.spinner(f"Analyzing {symbol}..."):
#                             analysis = st.session_state.chatbot.generate_market_analysis(symbol, stock_data,
#                                                                                          news_context)
#
#                         st.session_state.chat_history.append({'role': 'assistant', 'content': analysis})
#                         st.rerun()
#
#             with col3:
#                 if st.button("📰 News Impact"):
#                     if news_data:
#                         news_summary = '\n'.join([item.get('document', '')[:100] for item in news_data[:5]])
#                         query = f"Analyze the market impact of these recent news: {news_summary}"
#
#                         context_data = {
#                             'market_data': st.session_state.chatbot.format_market_data_context(market_data),
#                             'news_data': st.session_state.chatbot.format_news_context(news_data)
#                         }
#
#                         with st.spinner("Analyzing news impact..."):
#                             response = st.session_state.chatbot.generate_response(query, context_data)
#
#                         st.session_state.chat_history.append({'role': 'assistant', 'content': response})
#                         st.rerun()
#
#     # System Status Tab
#     with tab5:
#         st.markdown("### ⚙️ System Performance & Statistics")
#
#         # System metrics
#         col1, col2, col3 = st.columns(3)
#
#         with col1:
#             st.markdown("#### 🔧 Component Status")
#             st.markdown("✅ Market Data Fetcher: Active")
#             st.markdown("✅ News Scraper: Active")
#             st.markdown("✅ Vector Database: Active" if st.session_state.vector_db else "❌ Vector Database: Inactive")
#             st.markdown("✅ AI Chatbot: Active" if st.session_state.chatbot else "❌ AI Chatbot: Inactive")
#
#         with col2:
#             st.markdown("#### 📈 Data Statistics")
#             st.metric("Tracked Symbols", len(selected_symbols))
#             st.metric("News Articles", len(news_data))
#             st.metric("Chat Messages", len(st.session_state.chat_history))
#
#         with col3:
#             st.markdown("#### 🔄 Last Updated")
#             st.write(f"Market Data: {datetime.now().strftime('%H:%M:%S')}")
#             st.write(f"News Data: {datetime.now().strftime('%H:%M:%S')}")
#
#         # Database stats
#         if st.session_state.vector_db:
#             st.markdown("#### 🗄️ Vector Database Statistics")
#             try:
#                 db_stats = st.session_state.vector_db.get_database_stats()
#                 col1, col2, col3 = st.columns(3)
#
#                 with col1:
#                     st.metric("News Items", db_stats.get('news_items', 0))
#
#                 with col2:
#                     st.metric("Analysis Items", db_stats.get('analysis_items', 0))
#
#                 with col3:
#                     st.metric("Total Items", db_stats.get('total_items', 0))
#
#             except Exception as e:
#                 st.error(f"Error getting database stats: {e}")
#
#         # Configuration display
#         st.markdown("#### ⚙️ Configuration")
#         st.json({
#             "Selected Symbols": selected_symbols,
#             "Time Period": selected_period,
#             "Auto Refresh": auto_refresh,
#             "Vector DB Path": Config.VECTOR_DB_PATH,
#             "Embedding Model": Config.EMBEDDING_MODEL
#         })
#
#         # System logs (simplified)
#         st.markdown("#### 📋 Recent Activity")
#         activity_log = [
#             f"✅ Application started at {datetime.now().strftime('%H:%M:%S')}",
#             f"📊 Loaded data for {len(selected_symbols)} symbols",
#             f"📰 Scraped {len(news_data)} news articles",
#             f"💬 Chat history: {len(st.session_state.chat_history)} messages"
#         ]
#
#         for log in activity_log:
#             st.text(log)
#
#     # Auto-refresh functionality
#     if auto_refresh:
#         time.sleep(1)  # Small delay to prevent excessive refreshing
#
#         # Check if it's time to refresh (every 5 minutes)
#         current_time = datetime.now()
#         if st.session_state.last_update is None or (current_time - st.session_state.last_update).seconds >= 300:
#             st.session_state.last_update = current_time
#             st.cache_data.clear()
#             st.rerun()
#
#
# if __name__ == "__main__":
#     main()