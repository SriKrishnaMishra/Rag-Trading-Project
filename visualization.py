import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import streamlit as st


class FinancialVisualizer:
    """Create interactive financial charts and visualizations"""

    def __init__(self):
        self.default_colors = {
            'positive': '#00ff88',
            'negative': '#ff4444',
            'neutral': '#888888',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e'
        }

    def create_candlestick_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create interactive candlestick chart with technical indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price Action', 'Volume', 'RSI'),
            row_width=[0.2, 0.2, 0.7]
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color=self.default_colors['positive'],
                decreasing_line_color=self.default_colors['negative']
            ),
            row=1, col=1
        )

        # Add moving averages if available
        if 'MA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_20'],
                    mode='lines',
                    name='MA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

        if 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_50'],
                    mode='lines',
                    name='MA 50',
                    line=dict(color='purple', width=1)
                ),
                row=1, col=1
            )

        # Volume bars
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=self.default_colors['primary'],
                opacity=0.6
            ),
            row=2, col=1
        )

        # RSI if available
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )

            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    def create_portfolio_overview(self, market_summary: pd.DataFrame) -> go.Figure:
        """Create portfolio overview chart"""
        if market_summary.empty:
            return go.Figure().add_annotation(text="No data available")

        # Create subplot for multiple views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Changes', 'Volume Analysis', 'Performance Heatmap', 'Market Cap Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )

        # Price changes bar chart
        colors = [self.default_colors['positive'] if x > 0 else self.default_colors['negative']
                  for x in market_summary['Change%']]

        fig.add_trace(
            go.Bar(
                x=market_summary['Symbol'],
                y=market_summary['Change%'],
                marker_color=colors,
                name='Change %',
                text=[f"{x:.1f}%" for x in market_summary['Change%']],
                textposition='outside'
            ),
            row=1, col=1
        )

        # Volume analysis
        fig.add_trace(
            go.Bar(
                x=market_summary['Symbol'],
                y=market_summary['Volume'],
                marker_color=self.default_colors['secondary'],
                name='Volume'
            ),
            row=1, col=2
        )

        # Performance scatter plot
        fig.add_trace(
            go.Scatter(
                x=market_summary['Price'],
                y=market_summary['Change%'],
                mode='markers+text',
                text=market_summary['Symbol'],
                textposition='top center',
                marker=dict(
                    size=10,
                    color=market_summary['Change%'],
                    colorscale='RdYlGn',
                    showscale=True
                ),
                name='Price vs Change'
            ),
            row=2, col=1
        )

        # Market distribution pie chart
        fig.add_trace(
            go.Pie(
                labels=market_summary['Symbol'],
                values=market_summary['Volume'],
                name='Volume Distribution'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title='Market Overview Dashboard',
            template='plotly_dark',
            height=600,
            showlegend=False
        )

        return fig

    def create_news_sentiment_chart(self, news_data: List[Dict]) -> go.Figure:
        """Create news sentiment analysis chart"""
        if not news_data:
            return go.Figure().add_annotation(text="No news data available")

        # Extract sentiment data
        sentiments = [item.get('metadata', {}).get('sentiment', 'Neutral') for item in news_data]
        sentiment_counts = pd.Series(sentiments).value_counts()

        # Create pie chart
        colors = {
            'Positive': self.default_colors['positive'],
            'Negative': self.default_colors['negative'],
            'Neutral': self.default_colors['neutral']
        }

        fig = go.Figure(data=[
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker_colors=[colors.get(label, self.default_colors['neutral']) for label in sentiment_counts.index],
                textinfo='label+percent',
                textfont_size=12
            )
        ])

        fig.update_layout(
            title='News Sentiment Analysis',
            template='plotly_dark',
            height=400
        )

        return fig

    def create_technical_indicators_dashboard(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create comprehensive technical indicators dashboard"""
        if data.empty:
            return go.Figure().add_annotation(text=f"No data available for {symbol}")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MACD', 'Bollinger Bands', 'RSI', 'Volume Profile'),
            vertical_spacing=0.1
        )

        # MACD
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_signal'], name='Signal', line=dict(color='red')),
                row=1, col=1
            )

        # Bollinger Bands
        if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='white')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_upper'], name='Upper BB', line=dict(color='red', dash='dash')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_lower'], name='Lower BB', line=dict(color='green', dash='dash')),
                row=1, col=2
            )

        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

        # Volume Profile
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='orange'),
            row=2, col=2
        )

        fig.update_layout(
            title=f'{symbol} Technical Indicators Dashboard',
            template='plotly_dark',
            height=600,
            showlegend=False
        )

        return fig

    def create_comparison_chart(self, multiple_data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create comparison chart for multiple stocks"""
        if not multiple_data:
            return go.Figure().add_annotation(text="No comparison data available")

        fig = go.Figure()

        # Normalize prices to percentage change from first day
        colors = px.colors.qualitative.Set1

        for i, (symbol, data) in enumerate(multiple_data.items()):
            if not data.empty:
                # Calculate percentage change from first day
                pct_change = ((data['Close'] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=pct_change,
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )

        fig.update_layout(
            title='Stock Performance Comparison (% Change)',
            xaxis_title='Date',
            yaxis_title='Percentage Change (%)',
            template='plotly_dark',
            height=500,
            hovermode='x unified'
        )

        return fig

    def create_risk_metrics_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create risk metrics visualization"""
        if data.empty:
            return go.Figure().add_annotation(text=f"No data available for {symbol}")

        # Calculate risk metrics
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Create distribution plot
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Daily Returns',
                nbinsx=50,
                marker_color=self.default_colors['primary'],
                opacity=0.7
            )
        )

        # Add normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = np.exp(-0.5 * ((x_range - returns.mean()) / returns.std()) ** 2) / (
                    returns.std() * np.sqrt(2 * np.pi))
        normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 50  # Scale to match histogram

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2)
            )
        )

        fig.update_layout(
            title=f'{symbol} Risk Analysis - Daily Returns Distribution<br>Annualized Volatility: {volatility:.2%}',
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400
        )

        return fig



# -------------------------------------------------------------------
# # Enhanced AI Trading Analysis Hub with Advanced Global Market Data
# # This enhanced version includes support for Indian and international markets
# # with advanced analytics, AI insights, and real-time data processing
#
# import streamlit as st
# import pandas as pd
# import yfinance as yf
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta
# import requests
# from bs4 import BeautifulSoup
# import json
# from groq import Groq
# import time
# from typing import Dict, List, Optional
# import concurrent.futures
# from functools import lru_cache
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # Page Configuration
# st.set_page_config(
#     page_title="🚀 Advanced AI Trading Hub",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# # Enhanced CSS Styling
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 20px;
#         border-radius: 15px;
#         text-align: center;
#         color: white;
#         font-size: 2.5rem;
#         font-weight: bold;
#         margin-bottom: 30px;
#         box-shadow: 0 8px 32px rgba(0,0,0,0.1);
#     }
#
#     .metric-container {
#         background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
#         padding: 20px;
#         border-radius: 12px;
#         margin: 10px 0;
#         border-left: 5px solid #00ff88;
#         color: white;
#     }
#
#     .stock-card {
#         background: rgba(255, 255, 255, 0.05);
#         backdrop-filter: blur(10px);
#         border-radius: 15px;
#         padding: 20px;
#         margin: 10px 0;
#         border: 1px solid rgba(255, 255, 255, 0.1);
#     }
#
#     .news-item {
#         background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
#         color: #333;
#         padding: 15px;
#         border-radius: 10px;
#         margin: 10px 0;
#         border-left: 4px solid #ff6b6b;
#     }
#
#     .ai-response {
#         background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
#         color: #333;
#         padding: 20px;
#         border-radius: 12px;
#         margin: 15px 0;
#         border-left: 5px solid #667eea;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# class EnhancedMarketDataFetcher:
#     """Advanced market data fetcher for global markets including Indian stocks"""
#
#     def __init__(self):
#         self.session = requests.Session()
#         self.session.headers.update({
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         })
#
#     @lru_cache(maxsize=100)
#     def get_stock_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
#         """Enhanced stock data fetching with caching"""
#         try:
#             # Handle Indian stocks (.NS suffix) and international stocks
#             if not symbol.endswith('.NS') and not symbol.endswith('.BO') and len(symbol) <= 10:
#                 # Add .NS for Indian stocks if not already present
#                 test_symbol = f"{symbol}.NS"
#                 test_ticker = yf.Ticker(test_symbol)
#                 test_data = test_ticker.history(period="5d")
#
#                 if not test_data.empty:
#                     symbol = test_symbol
#
#             ticker = yf.Ticker(symbol)
#             data = ticker.history(period=period, interval=interval)
#
#             if data.empty:
#                 return pd.DataFrame()
#
#             # Add advanced technical indicators
#             data = self._add_advanced_indicators(data)
#             return data
#
#         except Exception as e:
#             st.error(f"Error fetching data for {symbol}: {e}")
#             return pd.DataFrame()
#
#     def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Add comprehensive technical indicators"""
#         try:
#             # Moving Averages
#             df['MA_10'] = df['Close'].rolling(window=10).mean()
#             df['MA_20'] = df['Close'].rolling(window=20).mean()
#             df['MA_50'] = df['Close'].rolling(window=50).mean()
#             df['MA_200'] = df['Close'].rolling(window=200).mean()
#
#             # Exponential Moving Averages
#             df['EMA_12'] = df['Close'].ewm(span=12).mean()
#             df['EMA_26'] = df['Close'].ewm(span=26).mean()
#
#             # RSI
#             df['RSI'] = self._calculate_rsi(df['Close'])
#
#             # MACD
#             df['MACD'] = df['EMA_12'] - df['EMA_26']
#             df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
#             df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
#
#             # Bollinger Bands
#             df['BB_Middle'] = df['Close'].rolling(window=20).mean()
#             bb_std = df['Close'].rolling(window=20).std()
#             df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
#             df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
#
#             # Stochastic Oscillator
#             low_14 = df['Low'].rolling(window=14).min()
#             high_14 = df['High'].rolling(window=14).max()
#             df['%K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
#             df['%D'] = df['%K'].rolling(window=3).mean()
#
#             # Average True Range (ATR)
#             df['TR'] = np.maximum(df['High'] - df['Low'],
#                                   np.maximum(abs(df['High'] - df['Close'].shift(1)),
#                                              abs(df['Low'] - df['Close'].shift(1))))
#             df['ATR'] = df['TR'].rolling(window=14).mean()
#
#             # Volume indicators
#             df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
#             df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
#
#             # Price momentum
#             df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
#
#             return df
#
#         except Exception as e:
#             st.error(f"Error adding indicators: {e}")
#             return df
#
#     def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
#         """Calculate RSI"""
#         delta = prices.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#         rs = gain / loss
#         rsi = 100 - (100 / (1 + rs))
#         return rsi
#
#     def get_indian_market_indices(self) -> Dict[str, pd.DataFrame]:
#         """Get major Indian market indices"""
#         indices = {
#             'NIFTY 50': '^NSEI',
#             'SENSEX': '^BSESN',
#             'NIFTY BANK': '^NSEBANK',
#             'NIFTY IT': '^CNXIT',
#             'NIFTY AUTO': '^CNXAUTO'
#         }
#
#         data = {}
#         for name, symbol in indices.items():
#             try:
#                 ticker_data = self.get_stock_data(symbol, period="1mo")
#                 if not ticker_data.empty:
#                     data[name] = ticker_data
#             except:
#                 continue
#
#         return data
#
#     def get_global_indices(self) -> Dict[str, pd.DataFrame]:
#         """Get major global indices"""
#         indices = {
#             'S&P 500': '^GSPC',
#             'NASDAQ': '^IXIC',
#             'DOW JONES': '^DJI',
#             'FTSE 100': '^FTSE',
#             'DAX': '^GDAXI',
#             'NIKKEI': '^N225',
#             'HANG SENG': '^HSI'
#         }
#
#         data = {}
#         for name, symbol in indices.items():
#             try:
#                 ticker_data = self.get_stock_data(symbol, period="1mo")
#                 if not ticker_data.empty:
#                     data[name] = ticker_data
#             except:
#                 continue
#
#         return data
#
#     def get_top_indian_stocks(self) -> List[str]:
#         """Get list of top Indian stocks"""
#         return [
#             'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
#             'ICICIBANK.NS', 'BHARTIARTL.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'LICI.NS',
#             'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'HCLTECH.NS', 'ASIANPAINT.NS',
#             'MARUTI.NS', 'AXISBANK.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS',
#             'WIPRO.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAMOTORS.NS', 'M&M.NS'
#         ]
#
#     def get_sector_analysis(self, symbols: List[str]) -> pd.DataFrame:
#         """Get sector-wise performance analysis"""
#         sector_data = []
#
#         # Define sectors (simplified mapping)
#         sector_mapping = {
#             'RELIANCE.NS': 'Oil & Gas', 'TCS.NS': 'IT', 'HDFCBANK.NS': 'Banking',
#             'INFY.NS': 'IT', 'ICICIBANK.NS': 'Banking', 'BHARTIARTL.NS': 'Telecom',
#             'SBIN.NS': 'Banking', 'ITC.NS': 'FMCG', 'MARUTI.NS': 'Auto',
#             'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
#             'TSLA': 'Auto', 'NVDA': 'Technology'
#         }
#
#         for symbol in symbols:
#             try:
#                 data = self.get_stock_data(symbol, period="5d")
#                 if not data.empty:
#                     current = data['Close'].iloc[-1]
#                     prev = data['Close'].iloc[0]
#                     change = ((current - prev) / prev) * 100
#
#                     sector_data.append({
#                         'Symbol': symbol,
#                         'Sector': sector_mapping.get(symbol, 'Other'),
#                         'Price': current,
#                         'Change%': change,
#                         'Volume': data['Volume'].iloc[-1]
#                     })
#             except:
#                 continue
#
#         return pd.DataFrame(sector_data)
#
#
# class EnhancedAIChatbot:
#     """Enhanced AI chatbot with improved model support"""
#
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#         self.client = Groq(api_key=api_key)
#         # ✅ UPDATED MODEL - This fixes your error!
#         self.model = "llama-3.1-70b-versatile"  # Updated to supported model
#         self.conversation_history = []
#
#     def analyze_stock(self, symbol: str, data: pd.DataFrame, news_context: str = "") -> str:
#         """Generate comprehensive stock analysis"""
#         if data.empty:
#             return f"No data available for {symbol} analysis."
#
#         latest = data.iloc[-1]
#         week_ago = data.iloc[-7] if len(data) >= 7 else data.iloc[0]
#
#         prompt = f"""
# Analyze {symbol} as a professional financial analyst:
#
# TECHNICAL DATA:
# - Current Price: ${latest['Close']:.2f}
# - 7-day Change: {((latest['Close'] - week_ago['Close']) / week_ago['Close'] * 100):.2f}%
# - RSI: {latest.get('RSI', 'N/A'):.2f}
# - MACD: {latest.get('MACD', 'N/A'):.4f}
# - Volume: {latest['Volume']:,.0f}
# - ATR: {latest.get('ATR', 'N/A'):.2f}
#
# NEWS CONTEXT: {news_context}
#
# Provide a comprehensive analysis covering:
# 1. Technical outlook (bullish/bearish signals)
# 2. Key support/resistance levels
# 3. Risk assessment
# 4. 1-week price target
# 5. Trading recommendation with rationale
#
# Keep analysis professional and data-driven.
# """
#
#         return self._get_ai_response(prompt)
#
#     def analyze_market_sentiment(self, market_data: Dict, global_indices: Dict) -> str:
#         """Analyze overall market sentiment"""
#         prompt = f"""
# Analyze current market sentiment based on this data:
#
# INDIAN MARKET:
# {self._format_market_summary(market_data)}
#
# GLOBAL MARKETS:
# {self._format_market_summary(global_indices)}
#
# Provide analysis on:
# 1. Overall market sentiment (bullish/bearish/neutral)
# 2. Key market drivers
# 3. Sector rotation trends
# 4. Risk factors to watch
# 5. Trading opportunities
#
# Focus on actionable insights for traders.
# """
#
#         return self._get_ai_response(prompt)
#
#     def _format_market_summary(self, data: Dict) -> str:
#         """Format market data for AI analysis"""
#         summary = ""
#         for name, df in data.items():
#             if not df.empty:
#                 current = df['Close'].iloc[-1]
#                 prev = df['Close'].iloc[0]
#                 change = ((current - prev) / prev) * 100
#                 summary += f"- {name}: ${current:.2f} ({change:+.2f}%)\n"
#         return summary
#
#     def _get_ai_response(self, prompt: str) -> str:
#         """Get response from AI model"""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system",
#                      "content": "You are an expert financial analyst providing professional market insights."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=1000,
#                 temperature=0.7
#             )
#
#             return response.choices[0].message.content
#
#         except Exception as e:
#             return f"AI Analysis temporarily unavailable: {str(e)}"
#
#
# class AdvancedVisualizer:
#     """Advanced visualization with multiple chart types"""
#
#     def create_advanced_dashboard(self, data: pd.DataFrame, symbol: str) -> go.Figure:
#         """Create comprehensive trading dashboard"""
#         fig = make_subplots(
#             rows=4, cols=2,
#             subplot_titles=(
#                 f'{symbol} - Price & Volume', 'Technical Indicators',
#                 'MACD Analysis', 'RSI & Stochastic',
#                 'Bollinger Bands', 'Volume Analysis',
#                 'Price Distribution', 'Momentum Indicators'
#             ),
#             specs=[[{"secondary_y": True}, {"secondary_y": False}],
#                    [{"secondary_y": False}, {"secondary_y": False}],
#                    [{"secondary_y": False}, {"secondary_y": False}],
#                    [{"secondary_y": False}, {"secondary_y": False}]],
#             vertical_spacing=0.08
#         )
#
#         # 1. Candlestick with Volume
#         fig.add_trace(
#             go.Candlestick(
#                 x=data.index, open=data['Open'], high=data['High'],
#                 low=data['Low'], close=data['Close'], name='Price',
#                 increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
#             ), row=1, col=1
#         )
#
#         # Add Moving Averages
#         for ma, color in [('MA_20', 'orange'), ('MA_50', 'purple')]:
#             if ma in data.columns:
#                 fig.add_trace(
#                     go.Scatter(x=data.index, y=data[ma], name=ma,
#                                line=dict(color=color, width=1)), row=1, col=1
#                 )
#
#         # Volume on secondary y-axis
#         fig.add_trace(
#             go.Bar(x=data.index, y=data['Volume'], name='Volume',
#                    marker_color='rgba(158,202,225,0.6)', yaxis='y2'), row=1, col=1
#         )
#
#         # 2. Technical Indicators Summary
#         if all(col in data.columns for col in ['RSI', 'MACD', '%K']):
#             indicators_df = data[['RSI', '%K']].iloc[-20:]  # Last 20 days
#             fig.add_trace(
#                 go.Scatter(x=indicators_df.index, y=indicators_df['RSI'],
#                            name='RSI', line=dict(color='purple')), row=1, col=2
#             )
#             fig.add_trace(
#                 go.Scatter(x=indicators_df.index, y=indicators_df['%K'],
#                            name='Stochastic %K', line=dict(color='blue')), row=1, col=2
#             )
#
#         # 3. MACD
#         if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
#             fig.add_trace(
#                 go.Scatter(x=data.index, y=data['MACD'], name='MACD',
#                            line=dict(color='blue')), row=2, col=1
#             )
#             fig.add_trace(
#                 go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal',
#                            line=dict(color='red')), row=2, col=1
#             )
#             fig.add_trace(
#                 go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram',
#                        marker_color='gray'), row=2, col=1
#             )
#
#         # 4. RSI with levels
#         if 'RSI' in data.columns:
#             fig.add_trace(
#                 go.Scatter(x=data.index, y=data['RSI'], name='RSI',
#                            line=dict(color='purple')), row=2, col=2
#             )
#             fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=2)
#             fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=2)
#
#         # 5. Bollinger Bands
#         if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
#             fig.add_trace(
#                 go.Scatter(x=data.index, y=data['Close'], name='Close',
#                            line=dict(color='white')), row=3, col=1
#             )
#             fig.add_trace(
#                 go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper',
#                            line=dict(color='red', dash='dash')), row=3, col=1
#             )
#             fig.add_trace(
#                 go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower',
#                            line=dict(color='green', dash='dash')), row=3, col=1
#             )
#
#         # 6. Volume Analysis
#         if 'Volume_Ratio' in data.columns:
#             fig.add_trace(
#                 go.Bar(x=data.index, y=data['Volume_Ratio'], name='Volume Ratio',
#                        marker_color='orange'), row=3, col=2
#             )
#
#         # 7. Price Distribution (Histogram)
#         fig.add_trace(
#             go.Histogram(x=data['Close'], name='Price Distribution',
#                          marker_color='lightblue', nbinsx=30), row=4, col=1
#         )
#
#         # 8. ROC (Rate of Change)
#         if 'ROC' in data.columns:
#             fig.add_trace(
#                 go.Scatter(x=data.index, y=data['ROC'], name='ROC',
#                            line=dict(color='orange')), row=4, col=2
#             )
#
#         fig.update_layout(
#             title=f'{symbol} - Advanced Technical Analysis Dashboard',
#             template='plotly_dark', height=1200, showlegend=False
#         )
#
#         return fig
#
#     def create_market_heatmap(self, sector_data: pd.DataFrame) -> go.Figure:
#         """Create market sector heatmap"""
#         if sector_data.empty:
#             return go.Figure()
#
#         # Pivot data for heatmap
#         heatmap_data = sector_data.pivot_table(
#             values='Change%', index='Sector', columns='Symbol', aggfunc='mean'
#         )
#
#         fig = go.Figure(data=go.Heatmap(
#             z=heatmap_data.values,
#             x=heatmap_data.columns,
#             y=heatmap_data.index,
#             colorscale='RdYlGn',
#             zmid=0,
#             text=heatmap_data.values,
#             texttemplate="%{text:.2f}%",
#             textfont={"size": 12},
#             hoverongaps=False
#         ))
#
#         fig.update_layout(
#             title='Market Sector Performance Heatmap',
#             template='plotly_dark',
#             height=500
#         )
#
#         return fig
#
#
# def main():
#     """Enhanced main application"""
#
#     # Header
#     st.markdown("""
#     <div class="main-header">
#         🚀 Advanced AI Trading Analysis Hub
#         <br><small>Global Markets • Indian Stocks • AI Insights</small>
#     </div>
#     """, unsafe_allow_html=True)
#
#     # Initialize session state
#     if 'market_data' not in st.session_state:
#         st.session_state.market_data = {}
#     if 'ai_chatbot' not in st.session_state:
#         st.session_state.ai_chatbot = None
#
#     # Sidebar Configuration
#     with st.sidebar:
#         st.markdown("### ⚙️ Advanced Configuration")
#
#         # API Key Input
#         groq_api_key = st.text_input("Groq API Key:", type="password",
#                                      help="Get free API key from https://console.groq.com")
#
#         if groq_api_key:
#             st.session_state.ai_chatbot = EnhancedAIChatbot(groq_api_key)
#             st.success("✅ AI Chatbot Connected")
#         else:
#             st.warning("⚠️ Enter Groq API key for AI features")
#
#         st.markdown("---")
#
#         # Market Selection
#         st.markdown("#### 🌍 Market Selection")
#         market_type = st.radio("Choose Market:",
#                                ["Indian Stocks", "US Stocks", "Global Indices", "Mixed Portfolio"])
#
#         # Stock Symbol Input
#         st.markdown("#### 📊 Stock Selection")
#
#         if market_type == "Indian Stocks":
#             fetcher = EnhancedMarketDataFetcher()
#             default_stocks = fetcher.get_top_indian_stocks()[:10]
#             selected_symbols = st.multiselect("Select Indian Stocks:",
#                                               default_stocks, default=default_stocks[:5])
#         elif market_type == "US Stocks":
#             us_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
#             selected_symbols = st.multiselect("Select US Stocks:",
#                                               us_stocks, default=us_stocks[:5])
#         elif market_type == "Global Indices":
#             selected_symbols = ['^NSEI', '^BSESN', '^GSPC', '^IXIC', '^FTSE']
#         else:  # Mixed Portfolio
#             indian_picks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
#             us_picks = ['AAPL', 'GOOGL', 'MSFT']
#             selected_symbols = indian_picks + us_picks
#
#         # Custom Symbol Input
#         custom_symbol = st.text_input("Add Custom Symbol:").upper()
#         if custom_symbol and st.button("Add Symbol"):
#             if custom_symbol not in selected_symbols:
#                 selected_symbols.append(custom_symbol)
#                 st.success(f"Added {custom_symbol}")
#
#         # Time Period
#         time_period = st.selectbox("Time Period:",
#                                    ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
#
#         # Refresh Controls
#         st.markdown("---")
#         auto_refresh = st.checkbox("Auto Refresh (30s)")
#         if st.button("🔄 Refresh All Data"):
#             st.cache_data.clear()
#             st.rerun()
#
#     # Main Content Tabs
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "📊 Market Overview",
#         "📈 Advanced Charts",
#         "🤖 AI Analysis",
#         "🌍 Global Markets",
#         "📱 Portfolio Tracker"
#     ])
#
#     # Initialize data fetcher
#     fetcher = EnhancedMarketDataFetcher()
#
#     with tab1:
#         st.markdown("### 📊 Real-Time Market Dashboard")
#
#         if selected_symbols:
#             # Create columns for metrics
#             col1, col2, col3, col4 = st.columns(4)
#
#             # Fetch market data
#             market_summary = fetcher.get_sector_analysis(selected_symbols)
#
#             if not market_summary.empty:
#                 avg_change = market_summary['Change%'].mean()
#                 gainers = len(market_summary[market_summary['Change%'] > 0])
#                 losers = len(market_summary[market_summary['Change%'] < 0])
#                 total_volume = market_summary['Volume'].sum()
#
#                 with col1:
#                     st.metric("Avg Change", f"{avg_change:.2f}%",
#                               delta=f"{avg_change:.2f}%")
#
#                 with col2:
#                     st.metric("Gainers", gainers, delta=gainers - losers)
#
#                 with col3:
#                     st.metric("Losers", losers)
#
#                 with col4:
#                     st.metric("Total Volume", f"{total_volume:,.0f}")
#
#                 # Market Summary Table
#                 st.markdown("#### 📋 Market Summary")
#                 styled_df = market_summary.style.format({
#                     'Price': '₹{:.2f}' if market_type == "Indian Stocks" else '${:.2f}',
#                     'Change%': '{:.2f}%',
#                     'Volume': '{:,.0f}'
#                 }).apply(lambda x: [
#                     'background-color: rgba(0,255,136,0.2)' if v > 0
#                     else 'background-color: rgba(255,68,68,0.2)' if v < 0
#                     else '' for v in x
#                 ], subset=['Change%'])
#
#                 st.dataframe(styled_df, use_container_width=True)
#
#                 # Sector Heatmap
#                 viz = AdvancedVisualizer()
#                 heatmap_fig = viz.create_market_heatmap(market_summary)
#                 st.plotly_chart(heatmap_fig, use_container_width=True)
#
#         else:
#             st.warning("Please select symbols to view market data")
#
#     with tab2:
#         st.markdown("### 📈 Advanced Technical Analysis")
#
#         if selected_symbols:
#             analysis_symbol = st.selectbox("Select for Analysis:", selected_symbols)
#
#             if analysis_symbol:
#                 with st.spinner(f"Loading advanced analysis for {analysis_symbol}..."):
#                     stock_data = fetcher.get_stock_data(analysis_symbol, time_period)
#
#                     if not stock_data.empty:
#                         viz = AdvancedVisualizer()
#                         dashboard_fig = viz.create_advanced_dashboard(stock_data, analysis_symbol)
#                         st.plotly_chart(dashboard_fig, use_container_width=True)
#
#                         # Technical Summary
#                         st.markdown("#### 📊 Technical Summary")
#                         latest = stock_data.iloc[-1]
#
#                         col1, col2, col3, col4 = st.columns(4)
#
#                         with col1:
#                             st.markdown(f"""
#                             <div class="metric-container">
#                                 <h4>Price Action</h4>
#                                 <p>Current: ₹{latest['Close']:.2f}</p>
#                                 <p>RSI: {latest.get('RSI', 0):.1f}</p>
#                             </div>
#                             """, unsafe_allow_html=True)
#
#                         with col2:
#                             macd_signal = "BUY" if latest.get('MACD', 0) > latest.get('MACD_Signal', 0) else "SELL"
#                             st.markdown(f"""
#                             <div class="metric-container">
#                                 <h4>MACD Signal</h4>
#                                 <p>Signal: {macd_signal}</p>
#                                 <p>MACD: {latest.get('MACD', 0):.4f}</p>
#                             </div>
#                             """, unsafe_allow_html=True)
#
#                         with col3:
#                             volume_signal = "HIGH" if latest.get('Volume_Ratio', 1) > 1.5 else "NORMAL"
#                             st.markdown(f"""
#                             <div class="metric-container">
#                                 <h4>Volume Analysis</h4>
#                                 <p>Status: {volume_signal}</p>
#                                 <p>Ratio: {latest.get('Volume_Ratio', 1):.2f}x</p>
#                             </div>
#                             """, unsafe_allow_html=True)
#
#                         with col4:
#                             atr_volatility = "HIGH" if latest.get('ATR', 0) > stock_data['ATR'].mean() else "NORMAL"
#                             st.markdown(f"""
#                             <div class="metric-container">
#                                 <h4>Volatility</h4>
#                                 <p>Level: {atr_volatility}</p>
#                                 <p>ATR: {latest.get('ATR', 0):.2f}</p>
#                             </div>
#                             """, unsafe_allow_html=True)
#                     else:
#                         st.error(f"No data available for {analysis_symbol}")
#         else:
#             st.warning("Please select symbols for technical analysis")
#
#     with tab3:
#         st.markdown("### 🤖 AI-Powered Market Analysis")
#
#         if st.session_state.ai_chatbot is None:
#             st.warning("⚠️ Please enter your Groq API key in the sidebar to enable AI features")
#         else:
#             # AI Analysis Options
#             col1, col2 = st.columns([1, 1])
#
#             with col1:
#                 st.markdown("#### 🎯 Quick AI Analysis")
#
#                 if st.button("📊 Analyze Selected Portfolio", type="primary"):
#                     if selected_symbols:
#                         with st.spinner("AI is analyzing your portfolio..."):
#                             # Get market data for analysis
#                             market_data = {}
#                             for symbol in selected_symbols[:5]:  # Limit to 5 for performance
#                                 data = fetcher.get_stock_data(symbol, "5d")
#                                 if not data.empty:
#                                     market_data[symbol] = data
#
#                             # Get global context
#                             global_indices = fetcher.get_global_indices()
#
#                             # Generate AI analysis
#                             analysis = st.session_state.ai_chatbot.analyze_market_sentiment(
#                                 market_data, global_indices
#                             )
#
#                             st.markdown(f"""
#                             <div class="ai-response">
#                                 <h4>🤖 AI Market Analysis</h4>
#                                 {analysis}
#                             </div>
#                             """, unsafe_allow_html=True)
#
#                 if st.button("🎯 Individual Stock Analysis"):
#                     if selected_symbols:
#                         stock_choice = st.selectbox("Choose stock for AI analysis:",
#                                                     selected_symbols, key="ai_stock_select")
#
#                         if stock_choice:
#                             with st.spinner(f"AI analyzing {stock_choice}..."):
#                                 stock_data = fetcher.get_stock_data(stock_choice, "1mo")
#
#                                 if not stock_data.empty:
#                                     analysis = st.session_state.ai_chatbot.analyze_stock(
#                                         stock_choice, stock_data, "Recent market volatility observed."
#                                     )
#
#                                     st.markdown(f"""
#                                     <div class="ai-response">
#                                         <h4>🎯 {stock_choice} AI Analysis</h4>
#                                         {analysis}
#                                     </div>
#                                     """, unsafe_allow_html=True)
#
#             with col2:
#                 st.markdown("#### 💬 Chat with AI Advisor")
#
#                 # Chat interface
#                 user_query = st.text_area("Ask anything about markets, stocks, or trading strategies:",
#                                           placeholder="E.g., Should I buy RELIANCE stock now? What's the outlook for IT sector?")
#
#                 if st.button("🚀 Get AI Insights") and user_query:
#                     with st.spinner("AI is thinking..."):
#                         # Create context from current market data
#                         context = f"Current portfolio: {', '.join(selected_symbols[:5])}\n"
#                         context += f"Market type: {market_type}\n"
#                         context += f"Time period: {time_period}\n"
#
#                         # Get AI response
#                         full_prompt = f"Context: {context}\n\nUser Question: {user_query}"
#                         response = st.session_state.ai_chatbot._get_ai_response(full_prompt)
#
#                         st.markdown(f"""
#                         <div class="ai-response">
#                             <h4>🤖 AI Advisor Response</h4>
#                             {response}
#                         </div>
#                         """, unsafe_allow_html=True)
#
#                 # Pre-built queries
#                 st.markdown("#### 🚀 Quick Questions")
#                 quick_queries = [
#                     "What are the top 3 stocks to watch today?",
#                     "Should I invest in IT sector now?",
#                     "What's the market outlook for next week?",
#                     "Which stocks have bullish technical signals?",
#                     "Analyze risk in current market conditions"
#                 ]
#
#                 for query in quick_queries:
#                     if st.button(query, key=f"quick_{query[:10]}"):
#                         with st.spinner("Getting AI insights..."):
#                             response = st.session_state.ai_chatbot._get_ai_response(query)
#                             st.markdown(f"""
#                             <div class="ai-response">
#                                 <h4>🤖 Quick Insight</h4>
#                                 {response}
#                             </div>
#                             """, unsafe_allow_html=True)
#
#     with tab4:
#         st.markdown("### 🌍 Global Markets Overview")
#
#         # Global Indices
#         col1, col2 = st.columns(2)
#
#         with col1:
#             st.markdown("#### 🇮🇳 Indian Market Indices")
#             indian_indices = fetcher.get_indian_market_indices()
#
#             if indian_indices:
#                 for name, data in indian_indices.items():
#                     if not data.empty:
#                         current = data['Close'].iloc[-1]
#                         prev = data['Close'].iloc[0]
#                         change = ((current - prev) / prev) * 100
#
#                         st.metric(name, f"₹{current:,.2f}", f"{change:+.2f}%")
#
#         with col2:
#             st.markdown("#### 🌍 Global Indices")
#             global_indices = fetcher.get_global_indices()
#
#             if global_indices:
#                 for name, data in global_indices.items():
#                     if not data.empty:
#                         current = data['Close'].iloc[-1]
#                         prev = data['Close'].iloc[0]
#                         change = ((current - prev) / prev) * 100
#
#                         st.metric(name, f"${current:,.2f}", f"{change:+.2f}%")
#
#         # Global Market Chart
#         st.markdown("#### 📈 Global Market Performance Comparison")
#
#         if global_indices:
#             fig = go.Figure()
#             colors = px.colors.qualitative.Set1
#
#             for i, (name, data) in enumerate(global_indices.items()):
#                 if not data.empty:
#                     # Normalize to percentage change from first day
#                     pct_change = ((data['Close'] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
#
#                     fig.add_trace(go.Scatter(
#                         x=data.index, y=pct_change, name=name,
#                         line=dict(color=colors[i % len(colors)], width=2)
#                     ))
#
#             fig.update_layout(
#                 title='Global Markets Performance Comparison (% Change)',
#                 xaxis_title='Date', yaxis_title='Percentage Change (%)',
#                 template='plotly_dark', height=500
#             )
#
#             st.plotly_chart(fig, use_container_width=True)
#
#     with tab5:
#         st.markdown("### 📱 Advanced Portfolio Tracker")
#
#         if selected_symbols:
#             # Portfolio Summary
#             st.markdown("#### 💼 Portfolio Performance")
#
#             portfolio_data = []
#             total_investment = 0
#
#             for symbol in selected_symbols:
#                 # Simulate investment amounts (in real app, user would input these)
#                 investment_amount = np.random.randint(10000, 100000)
#
#                 data = fetcher.get_stock_data(symbol, "1mo")
#                 if not data.empty:
#                     current_price = data['Close'].iloc[-1]
#                     entry_price = data['Close'].iloc[0]
#
#                     shares = investment_amount / entry_price
#                     current_value = shares * current_price
#                     profit_loss = current_value - investment_amount
#                     profit_loss_pct = (profit_loss / investment_amount) * 100
#
#                     portfolio_data.append({
#                         'Symbol': symbol,
#                         'Investment': investment_amount,
#                         'Current Value': current_value,
#                         'P&L': profit_loss,
#                         'P&L %': profit_loss_pct,
#                         'Shares': shares
#                     })
#
#                     total_investment += investment_amount
#
#             if portfolio_data:
#                 portfolio_df = pd.DataFrame(portfolio_data)
#                 total_current_value = portfolio_df['Current Value'].sum()
#                 total_pnl = total_current_value - total_investment
#                 total_pnl_pct = (total_pnl / total_investment) * 100
#
#                 # Portfolio Metrics
#                 col1, col2, col3, col4 = st.columns(4)
#
#                 with col1:
#                     st.metric("Total Investment", f"₹{total_investment:,.0f}")
#
#                 with col2:
#                     st.metric("Current Value", f"₹{total_current_value:,.0f}")
#
#                 with col3:
#                     st.metric("Total P&L", f"₹{total_pnl:,.0f}", f"{total_pnl_pct:+.2f}%")
#
#                 with col4:
#                     best_performer = portfolio_df.loc[portfolio_df['P&L %'].idxmax(), 'Symbol']
#                     st.metric("Best Performer", best_performer)
#
#                 # Portfolio Table
#                 st.markdown("#### 📊 Detailed Portfolio Breakdown")
#                 styled_portfolio = portfolio_df.style.format({
#                     'Investment': '₹{:,.0f}',
#                     'Current Value': '₹{:,.0f}',
#                     'P&L': '₹{:,.0f}',
#                     'P&L %': '{:.2f}%',
#                     'Shares': '{:.2f}'
#                 }).apply(lambda x: [
#                     'background-color: rgba(0,255,136,0.2)' if v > 0
#                     else 'background-color: rgba(255,68,68,0.2)' if v < 0
#                     else '' for v in x
#                 ], subset=['P&L', 'P&L %'])
#
#                 st.dataframe(styled_portfolio, use_container_width=True)
#
#                 # Portfolio Allocation Chart
#                 fig_pie = px.pie(
#                     portfolio_df, values='Current Value', names='Symbol',
#                     title='Portfolio Allocation by Value'
#                 )
#                 fig_pie.update_layout(template='plotly_dark', height=400)
#
#                 # Performance Chart
#                 fig_bar = px.bar(
#                     portfolio_df, x='Symbol', y='P&L %',
#                     color='P&L %', color_continuous_scale='RdYlGn',
#                     title='Individual Stock Performance (%)'
#                 )
#                 fig_bar.update_layout(template='plotly_dark', height=400)
#
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.plotly_chart(fig_pie, use_container_width=True)
#                 with col2:
#                     st.plotly_chart(fig_bar, use_container_width=True)
#
#                 # Risk Analysis
#                 st.markdown("#### ⚠️ Portfolio Risk Analysis")
#
#                 returns_data = []
#                 for symbol in selected_symbols:
#                     data = fetcher.get_stock_data(symbol, "3mo")
#                     if not data.empty:
#                         daily_returns = data['Close'].pct_change().dropna()
#                         volatility = daily_returns.std() * np.sqrt(252)  # Annualized
#                         returns_data.append({
#                             'Symbol': symbol,
#                             'Volatility': volatility,
#                             'Avg Daily Return': daily_returns.mean(),
#                             'Max Drawdown': (daily_returns.cumsum().expanding().max() - daily_returns.cumsum()).max()
#                         })
#
#                 if returns_data:
#                     risk_df = pd.DataFrame(returns_data)
#
#                     col1, col2 = st.columns(2)
#
#                     with col1:
#                         # Volatility comparison
#                         fig_vol = px.bar(
#                             risk_df, x='Symbol', y='Volatility',
#                             title='Volatility Comparison (Annualized)',
#                             color='Volatility', color_continuous_scale='Reds'
#                         )
#                         fig_vol.update_layout(template='plotly_dark', height=400)
#                         st.plotly_chart(fig_vol, use_container_width=True)
#
#                     with col2:
#                         # Risk-Return Scatter
#                         fig_scatter = px.scatter(
#                             risk_df, x='Volatility', y='Avg Daily Return',
#                             size='Max Drawdown', hover_data=['Symbol'],
#                             title='Risk-Return Profile',
#                             color='Avg Daily Return', color_continuous_scale='RdYlGn'
#                         )
#                         fig_scatter.update_layout(template='plotly_dark', height=400)
#                         st.plotly_chart(fig_scatter, use_container_width=True)
#
#         else:
#             st.warning("Select stocks to track your portfolio")
#
#     # Auto-refresh functionality
#     if auto_refresh:
#         time.sleep(30)  # Refresh every 30 seconds
#         st.rerun()
#
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center; color: #888888; padding: 20px;'>
#         <p>🚀 Advanced AI Trading Analysis Hub | Powered by Groq AI & Yahoo Finance</p>
#         <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#
# if __name__ == "__main__":
#     main()




# ------------------------------------------------------------------------------------
