# 🚀 AI Trading Analysis Hub


Main Topic any one run this project first create .env file 
then add : - this api key

# API Keys for Financial Trading Analysis Hub
# Copy this file to .env and add your actual API keys

# REQUIRED: Groq API Key (Free tier available at https://console.groq.com)
# Replace 'your_groq_api_key_here' with your actual API key from Groq
GROQ_API_KEY=your_groq_api_key_here

# OPTIONAL: News API Key (Free tier at https://newsapi.org)
# Not required as the app uses free RSS feeds as fallback
NEWS_API_KEY=

# OPTIONAL: Alpha Vantage API Key (Free tier available)
# Currently not used but can be integrated for additional data
ALPHA_VANTAGE_KEY=

# Application Settings (don't change these unless you know what you're doing)
VECTOR_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# SETUP INSTRUCTIONS:
# ===================
# 1. Get a FREE Groq API key:
#    - Visit: https://console.groq.com
#    - Sign up (no credit card required)
#    - Create an API key
#
# 2. Replace the line above:
#    Change: GROQ_API_KEY=your_groq_api_key_here
#    To:     GROQ_API_KEY=gsk_your_actual_key_here
#
# 3. Save this file as '.env' (remove the .example part)
#
# 4. Test your setup by running: python test_groq_api.py
#
# Your API key should look something like:
# gsk_1234567890abcdef1234567890abcdef1234567890abcdef
#
# Keep your API key secret - never share it publicly!


-----------------------------------------------------------------------------------------------------------------------------------

A comprehensive, free, and open-source trading analysis platform with AI chatbot integration powered by Groq's free API.

## 🎯 Features

- **Real-time Market Data**: Live stock prices, technical indicators, volume analysis
- **AI-Powered Chatbot**: Financial advisor using Groq's free API
- **News Sentiment Analysis**: Automatic news scraping with sentiment scoring
- **Advanced Visualizations**: Interactive charts and technical analysis
- **Vector Database**: ChromaDB for intelligent data storage and retrieval
- **Free & Open Source**: No paid subscriptions required

## 📋 Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection for real-time data
- Groq API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SriKrishnaMishra/Rag-Trading-Project.git
   cd Rag-Trading-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env file with your API keys
   # See Environment Variables section below
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

The application will open at `http://localhost:8501`

## 🔑 Environment Variables

**IMPORTANT**: Create a `.env` file in the project root with the following variables:

```bash
# Required - Get your free API key from https://console.groq.com
GROQ_API_KEY=your_actual_groq_api_key_here

# Optional - Get from https://newsapi.org (free tier available)
NEWS_API_KEY=your_news_api_key_here
```

### How to get API keys:

1. **Groq API (Required - Free)**
   - Visit [https://console.groq.com](https://console.groq.com)
   - Sign up for a free account
   - Create an API key
   - Copy the key to your `.env` file

2. **News API (Optional - Free tier available)**
   - Visit [https://newsapi.org](https://newsapi.org)
   - Sign up for free account (28 days trial, then limited free tier)
   - Get your API key
   - This is optional - the app uses free RSS feeds as fallback

## 📁 Project Structure

```
rag-project-2-trading/
├── main.py                 # Main Streamlit application
├── config.py              # Configuration management
├── data_fetcher.py         # Market data collection
├── news_scraper.py         # News scraping and sentiment analysis
├── vector_database.py      # Vector database management
├── groq_chatbot.py         # AI chatbot integration
├── visualization.py        # Charts and graphs
├── requirements.txt        # Python dependencies
├── requirements_fallback.txt # Fallback dependencies
├── install_dependencies.py # Dependency installation script
├── test_groq_api.py       # API testing script
├── run_app.py             # Alternative app runner
├── .env.example           # Environment variables template
├── .env                   # Your actual environment variables (not in git)
├── .gitignore            # Git ignore rules
├── SETUP.md              # Detailed setup guide
└── README.md             # This file
```

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

## 🔒 Security Notes

- **Never commit your `.env` file** to version control
- Keep API keys secure and don't share them
- The application runs locally - your data stays on your machine
- No personal information is sent to external services

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
1. Check this README and SETUP.md first
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

## 📚 Additional Documentation

- [SETUP.md](SETUP.md) - Detailed setup guide with advanced configuration
- [Requirements](requirements.txt) - Complete list of Python dependencies

---

**Enjoy your new AI-powered trading analysis platform! 🎉** 
