# 🚀 AI Trading Analysis Hub - Setup Guide

A comprehensive, free, and open-source trading analysis platform with AI chatbot integration.

## 🎯 Features

- **Real-time Market Data**: Live stock prices, technical indicators, volume analysis
- **AI-Powered Chatbot**: Financial advisor using Groq's free API
- **News Sentiment Analysis**: Automatic news scraping with sentiment scoring
- **Advanced Visualizations**: Interactive charts and technical analysis
- **Vector Database**: ChromaDB for intelligent data storage and retrieval
- **Free & Open Source**: No paid subscriptions required

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection for real-time data
- Groq API key (free tier available)

## 🛠️ Installation Steps

### 1. Clone or Download the Project

Create a new folder and save all the provided Python files:

```
trading_hub/
├── main.py                 # Main Streamlit application
├── config.py              # Configuration management
├── data_fetcher.py         # Market data collection
├── news_scraper.py         # News scraping and sentiment analysis
├── vector_database.py      # Vector database management
├── groq_chatbot.py         # AI chatbot integration
├── visualization.py        # Charts and graphs
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── SETUP.md               # This setup guide
```

### 2. Install Python Dependencies

Open terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

### 3. Get Free API Keys

#### Groq API (Required - Free)
1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Create an API key
4. Copy the API key for the next step

#### News API (Optional - Free tier available)
1. Visit [https://newsapi.org](https://newsapi.org)
2. Sign up for free account (28 days trial, then limited free tier)
3. Get your API key
4. This is optional - the app uses free RSS feeds as fallback

### 4. Configure Environment Variables

1. Copy `.env.example` to `.env`
2. Open `.env` file and add your API keys:

```bash
# Required
GROQ_API_KEY=your_actual_groq_api_key_here

# Optional
NEWS_API_KEY=your_news_api_key_here
```

### 5. Run the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 🎮 How to Use

### Market Overview
- View real-time stock prices and changes
- Monitor portfolio performance
- See market gainers and losers
- Interactive portfolio visualization

### Technical Analysis
- Detailed candlestick charts with indicators
- RSI, MACD, Bollinger Bands, Moving Averages
- Risk analysis and volatility metrics
- Multi-stock comparison charts

### News & Sentiment
- Real-time financial news aggregation
- Automatic sentiment analysis (Positive/Negative/Neutral)
- News impact on market movements
- Source tracking and timestamping

### AI Chatbot
- Ask questions about market trends
- Get stock analysis and recommendations
- Trading strategy discussions
- News impact analysis
- Context-aware responses using market data

### System Status
- Monitor all system components
- Database statistics and health
- Performance metrics
- Configuration overview

## 🔧 Customization

### Adding New Stocks
- Use the sidebar to add/remove symbols
- Supports any valid stock ticker
- Real-time data fetching

### Modifying Time Periods
- 1 day to 1 year historical data
- Adjustable via sidebar interface
- Affects all technical indicators

### Adding New Data Sources
Edit `config.py` to add new RSS feeds:
```python
NEWS_SOURCES = [
    'https://feeds.finance.yahoo.com/rss/2.0/headline',
    'https://your-custom-feed.com/rss',
    # Add more sources here
]
```

### Extending AI Capabilities
Modify `groq_chatbot.py` to add new analysis functions or change the AI model.

## 📊 Technical Architecture

### Data Sources (All Free)
- **Yahoo Finance**: Stock prices, volume, historical data
- **RSS Feeds**: Financial news from Reuters, Yahoo, CNN
- **yfinance**: Python library for Yahoo Finance API
- **Technical Analysis**: Using TA-Lib indicators

### AI Integration
- **Groq**: Fast, free inference for LLMs
- **Mixtral 8x7B**: High-performance language model
- **Context-aware**: Uses real-time market data in responses

### Vector Database
- **ChromaDB**: Open-source vector database
- **Sentence Transformers**: Text embeddings
- **Persistent Storage**: Local database files

### Visualization
- **Plotly**: Interactive charts and graphs
- **Streamlit**: Web application framework
- **Real-time Updates**: Auto-refresh capabilities

## 🚨 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found" error**
   - Make sure you created the `.env` file
   - Check that your API key is correct
   - Ensure no extra spaces in the .env file

2. **Import errors**
   - Run `pip install -r requirements.txt` again
   - Check Python version (3.8+ required)
   - Try `pip install --upgrade streamlit`

3. **No market data showing**
   - Check internet connection
   - Try different stock symbols
   - Yahoo Finance may have temporary outages

4. **Charts not displaying**
   - Update browsers (Chrome/Firefox recommended)
   - Clear browser cache
   - Check JavaScript is enabled

5. **Slow performance**
   - Reduce number of stocks being analyzed
   - Increase cache TTL in main.py
   - Close other browser tabs

### Performance Optimization

1. **Reduce Data Load**
   ```python
   # In main.py, increase cache time
   @st.cache_data(ttl=600)  # 10 minutes instead of 5
   ```

2. **Limit News Sources**
   ```python
   # In config.py, reduce news sources
   NEWS_SOURCES = [
       'https://feeds.finance.yahoo.com/rss/2.0/headline'
       # Comment out others for faster loading
   ]
   ```

3. **Database Maintenance**
   - Clear old vector database files periodically
   - Restart application weekly for best performance

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep API keys secure and don't share them
- The application runs locally - your data stays on your machine
- No personal information is sent to external services

## 📈 Advanced Features

### Custom Indicators
Add your own technical indicators in `data_fetcher.py`:
```python
def _add_custom_indicator(self, df):
    # Your custom indicator logic here
    df['Custom_Indicator'] = your_calculation
    return df
```

### News Source Integration
Add new news sources in `news_scraper.py`:
```python
def scrape_custom_source(self, url):
    # Your custom scraping logic
    return news_articles
```

### AI Model Switching
Change the AI model in `groq_chatbot.py`:
```python
self.model = "llama2-70b-4096"  # Different model
```

## 💡 Tips for Best Experience

1. **Start Small**: Begin with 3-5 stocks for better performance
2. **Regular Updates**: Enable auto-refresh for real-time data
3. **Ask Specific Questions**: The AI works better with detailed queries
4. **Use Context**: Reference specific stocks or timeframes in chat
5. **Monitor Resources**: Check system status tab for performance

## 🚀 Future Enhancements

This is a foundation that can be extended with:
- Options and futures data integration
- Portfolio optimization algorithms
- Backtesting capabilities
- Alert system for price movements
- Email/SMS notifications
- Mobile-responsive design
- Multi-language support

## 🤝 Contributing

This is an open-source project. Feel free to:
- Add new features
- Fix bugs
- Improve documentation
- Share feedback and suggestions

## 📞 Support

If you encounter issues:
1. Check this setup guide first
2. Review error messages carefully
3. Ensure all API keys are configured
4. Try restarting the application
5. Check system requirements

## 🏆 Success Indicators

You'll know the setup is successful when you see:
- ✅ Green status indicators in the sidebar
- 📊 Real-time stock data loading
- 📰 News articles appearing with sentiment
- 🤖 AI chatbot responding to questions
- 📈 Interactive charts rendering properly

Enjoy your new AI-powered trading analysis platform! 🎉