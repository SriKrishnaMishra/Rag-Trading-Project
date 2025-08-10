import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Try to import TA-Lib, fall back to manual calculations if not available
try:
    import talib as ta

    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    print("TA-Lib not available, using manual calculations for technical indicators")


class MarketDataFetcher:
    """Fetch real-time and historical market data using free APIs"""

    def __init__(self):
        self.session = requests.Session()

    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch stock data using yfinance (free)"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                return pd.DataFrame()

            # Add technical indicators
            data = self._add_technical_indicators(data)
            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators with fallback calculations"""
        try:
            if TA_LIB_AVAILABLE:
                return self._add_talib_indicators(df)
            else:
                return self._add_manual_indicators(df)
        except Exception as e:
            print(f"Error adding indicators: {e}")
            return df

    def _add_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators using TA-Lib"""
        # Moving averages
        df['MA_20'] = ta.SMA(df['Close'].values, timeperiod=20)
        df['MA_50'] = ta.SMA(df['Close'].values, timeperiod=50)

        # RSI
        df['RSI'] = ta.RSI(df['Close'].values, timeperiod=14)

        # MACD
        df['MACD'], df['MACD_signal'], _ = ta.MACD(df['Close'].values)

        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(df['Close'].values, timeperiod=20)

        return df

    def _add_manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators using manual calculations"""
        # Moving averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # RSI calculation
        df['RSI'] = self._calculate_rsi(df['Close'])

        # MACD calculation
        df['MACD'], df['MACD_signal'] = self._calculate_macd(df['Close'])

        # Bollinger Bands
        df['BB_upper'], df['BB_lower'] = self._calculate_bollinger_bands(df['Close'])

        return df

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD manually"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands manually"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, lower_band

    def get_multiple_stocks(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        data = {}
        for symbol in symbols:
            stock_data = self.get_stock_data(symbol, period)
            if not stock_data.empty:
                data[symbol] = stock_data
        return data

    def get_market_summary(self, symbols: List[str]) -> pd.DataFrame:
        """Get current market summary"""
        summary_data = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100

                    summary_data.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'Change': change,
                        'Change%': change_pct,
                        'Volume': hist['Volume'].iloc[-1],
                        'Name': info.get('longName', symbol)
                    })

            except Exception as e:
                print(f"Error getting summary for {symbol}: {e}")
                continue

        return pd.DataFrame(summary_data)

    def get_trending_stocks(self) -> List[str]:
        """Get trending stocks (free method using predefined list)"""
        # This would normally use an API, but for free version,
        # we'll return popular stocks
        trending = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL',
                    'AMZN', 'META', 'NFLX', 'AMD', 'INTC']
        return trending[:5]  # Return top 5