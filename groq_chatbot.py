from groq import Groq
import json
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from config import Config


class FinancialChatbot:
    """AI-powered financial chatbot using Groq API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.get_groq_key()
        self.client = Groq(api_key=self.api_key)
        self.conversation_history = []
        self.model = "mixtral-8x7b-32768"  # Free model with good performance

    def format_market_data_context(self, market_data: Dict) -> str:
        """Format market data for chatbot context"""
        context = "Current Market Data:\n"

        for symbol, data in market_data.items():
            if not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest

                change = latest['Close'] - prev['Close']
                change_pct = (change / prev['Close']) * 100

                context += f"""
{symbol}:
- Price: ${latest['Close']:.2f}
- Change: ${change:.2f} ({change_pct:.2f}%)
- Volume: {latest['Volume']:,.0f}
- RSI: {latest.get('RSI', 0):.2f}
- 20-day MA: ${latest.get('MA_20', 0):.2f}
"""

        return context

    def format_news_context(self, news_data: List[Dict]) -> str:
        """Format news data for chatbot context"""
        if not news_data:
            return "No recent news available."

        context = "Recent Financial News:\n"

        for item in news_data[:5]:  # Top 5 news items
            metadata = item.get('metadata', {})
            context += f"""
- {metadata.get('source', 'Unknown')}: {item.get('document', '')[:150]}...
  Sentiment: {metadata.get('sentiment', 'Neutral')}
  Time: {metadata.get('timestamp', 'Unknown')}
"""

        return context

    def create_system_prompt(self, context_data: Dict = None) -> str:
        """Create system prompt with market context"""
        base_prompt = """You are an expert financial analyst and trading advisor. You provide clear, actionable insights about stocks, market trends, and trading strategies.

Key guidelines:
- Provide specific, data-driven analysis
- Include both technical and fundamental perspectives
- Mention risk factors and potential opportunities
- Use clear, professional language
- Always include disclaimers about investment risks
- Focus on educational content, not direct investment advice

Current market context will be provided with each query."""

        if context_data:
            market_context = context_data.get('market_data', '')
            news_context = context_data.get('news_data', '')

            base_prompt += f"\n\nCURRENT MARKET DATA:\n{market_context}"
            base_prompt += f"\n\nRECENT NEWS:\n{news_context}"

        return base_prompt

    def generate_response(self, user_query: str, context_data: Dict = None) -> str:
        """Generate AI response using Groq"""
        try:
            system_prompt = self.create_system_prompt(context_data)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            # Add conversation history (last 4 messages)
            if self.conversation_history:
                for msg in self.conversation_history[-4:]:
                    messages.insert(-1, msg)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9
            )

            ai_response = response.choices[0].message.content

            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": ai_response}
            ])

            return ai_response

        except Exception as e:
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

    def generate_market_analysis(self, symbol: str, stock_data: pd.DataFrame, news_context: str = "") -> str:
        """Generate detailed market analysis for a specific stock"""
        if stock_data.empty:
            return f"No data available for {symbol} analysis."

        # Prepare data summary
        latest = stock_data.iloc[-1]
        prev_week = stock_data.iloc[-7] if len(stock_data) >= 7 else stock_data.iloc[0]

        analysis_prompt = f"""
Analyze {symbol} based on the following data:

Technical Indicators:
- Current Price: ${latest['Close']:.2f}
- 7-day change: {((latest['Close'] - prev_week['Close']) / prev_week['Close'] * 100):.2f}%
- RSI: {latest.get('RSI', 'N/A')}
- 20-day MA: ${latest.get('MA_20', 0):.2f}
- 50-day MA: ${latest.get('MA_50', 0):.2f}
- Volume: {latest['Volume']:,.0f}

News Context:
{news_context}

Please provide:
1. Technical analysis summary
2. Key support/resistance levels
3. Short-term outlook (1-2 weeks)
4. Risk factors
5. Key catalysts to watch

Keep the analysis concise but comprehensive.
"""

        return self.generate_response(analysis_prompt)

    def get_trading_insights(self, market_summary: pd.DataFrame, news_data: List[Dict]) -> str:
        """Generate trading insights based on market data"""
        market_overview = ""
        if not market_summary.empty:
            gainers = market_summary[market_summary['Change%'] > 0].head(3)
            losers = market_summary[market_summary['Change%'] < 0].head(3)

            market_overview = f"""
Market Overview:
Top Gainers: {', '.join(gainers['Symbol'].tolist()) if not gainers.empty else 'None'}
Top Losers: {', '.join(losers['Symbol'].tolist()) if not losers.empty else 'None'}
Average Change: {market_summary['Change%'].mean():.2f}%
"""

        news_summary = ""
        if news_data:
            positive_news = sum(1 for item in news_data if item.get('metadata', {}).get('sentiment') == 'Positive')
            negative_news = sum(1 for item in news_data if item.get('metadata', {}).get('sentiment') == 'Negative')

            news_summary = f"""
News Sentiment:
- Positive: {positive_news}
- Negative: {negative_news}
- Neutral: {len(news_data) - positive_news - negative_news}
"""

        insight_prompt = f"""
Based on current market conditions, provide trading insights:

{market_overview}
{news_summary}

Please provide:
1. Market sentiment analysis
2. Sector trends to watch
3. Risk assessment
4. 3 key trading opportunities
5. Market outlook for next week

Focus on actionable insights for traders.
"""

        return self.generate_response(insight_prompt)

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_conversation_summary(self) -> str:
        """Get summary of current conversation"""
        if not self.conversation_history:
            return "No conversation history available."

        summary = f"Conversation contains {len(self.conversation_history)} messages.\n"
        summary += f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"

        return summary