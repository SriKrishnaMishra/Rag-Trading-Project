import requests
import feedparser
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import re


class NewsDataScraper:
    """Scrape financial news from free sources"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_yahoo_finance_news(self, symbol: str) -> List[Dict]:
        """Scrape news from Yahoo Finance for specific symbol"""
        news_articles = []
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find news articles (Yahoo's structure might change)
            articles = soup.find_all('h3', class_=re.compile('Mb\(5px\)'))

            for article in articles[:5]:  # Get top 5 articles
                try:
                    title = article.get_text().strip()
                    link_element = article.find('a')
                    link = link_element.get('href') if link_element else ''

                    if title and len(title) > 10:
                        news_articles.append({
                            'title': title,
                            'link': link,
                            'source': 'Yahoo Finance',
                            'symbol': symbol,
                            'timestamp': datetime.now()
                        })
                except Exception as e:
                    continue

        except Exception as e:
            print(f"Error scraping Yahoo Finance for {symbol}: {e}")

        return news_articles

    def get_rss_news(self, rss_urls: List[str]) -> List[Dict]:
        """Get news from RSS feeds"""
        all_news = []

        for url in rss_urls:
            try:
                feed = feedparser.parse(url)

                for entry in feed.entries[:10]:  # Get top 10 from each feed
                    published = getattr(entry, 'published_parsed', None)
                    if published:
                        pub_date = datetime(*published[:6])
                    else:
                        pub_date = datetime.now()

                    # Only get recent news (last 7 days)
                    if (datetime.now() - pub_date).days <= 7:
                        all_news.append({
                            'title': entry.title,
                            'link': entry.link,
                            'summary': getattr(entry, 'summary', ''),
                            'source': feed.feed.get('title', 'RSS Feed'),
                            'timestamp': pub_date
                        })

            except Exception as e:
                print(f"Error parsing RSS feed {url}: {e}")
                continue

        return sorted(all_news, key=lambda x: x['timestamp'], reverse=True)

    def get_market_news_summary(self, symbols: List[str]) -> pd.DataFrame:
        """Get comprehensive news summary for symbols"""
        all_news = []

        # Get RSS news
        rss_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.reuters.com/reuters/businessNews'
        ]

        rss_news = self.get_rss_news(rss_sources)
        all_news.extend(rss_news)

        # Get symbol-specific news
        for symbol in symbols[:3]:  # Limit to avoid rate limiting
            symbol_news = self.scrape_yahoo_finance_news(symbol)
            all_news.extend(symbol_news)

        if not all_news:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_news)
        df = df.drop_duplicates(subset=['title'])
        df = df.sort_values('timestamp', ascending=False)

        return df.head(20)  # Return top 20 news items

    def analyze_sentiment_simple(self, text: str) -> str:
        """Simple sentiment analysis using keyword matching"""
        positive_words = ['bull', 'bullish', 'gain', 'rise', 'up', 'growth',
                          'profit', 'surge', 'rally', 'positive', 'strong']
        negative_words = ['bear', 'bearish', 'loss', 'fall', 'down', 'decline',
                          'drop', 'crash', 'negative', 'weak', 'sell']

        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)

        if pos_count > neg_count:
            return 'Positive'
        elif neg_count > pos_count:
            return 'Negative'
        else:
            return 'Neutral'

    def get_news_with_sentiment(self, symbols: List[str]) -> pd.DataFrame:
        """Get news with basic sentiment analysis"""
        news_df = self.get_market_news_summary(symbols)

        if not news_df.empty:
            news_df['sentiment'] = news_df['title'].apply(self.analyze_sentiment_simple)
            if 'summary' in news_df.columns:
                news_df['summary_sentiment'] = news_df['summary'].apply(self.analyze_sentiment_simple)

        return news_df