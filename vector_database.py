import chromadb
from chromadb.config import Settings
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
from datetime import datetime
import os


class VectorDatabase:
    """Manage vector database for financial data and news"""

    def __init__(self, db_path: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create collections
        self._init_collections()

    def _init_collections(self):
        """Initialize database collections"""
        try:
            self.news_collection = self.client.get_or_create_collection(
                name="financial_news",
                metadata={"hnsw:space": "cosine"}
            )

            self.market_collection = self.client.get_or_create_collection(
                name="market_data",
                metadata={"hnsw:space": "cosine"}
            )

            self.analysis_collection = self.client.get_or_create_collection(
                name="market_analysis",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error initializing collections: {e}")

    def add_news_data(self, news_df: pd.DataFrame):
        """Add news data to vector database"""
        if news_df.empty:
            return

        documents = []
        metadatas = []
        ids = []

        for idx, row in news_df.iterrows():
            # Combine title and summary for better context
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            documents.append(text)

            metadata = {
                'source': str(row.get('source', '')),
                'timestamp': str(row.get('timestamp', datetime.now())),
                'symbol': str(row.get('symbol', '')),
                'sentiment': str(row.get('sentiment', 'Neutral')),
                'link': str(row.get('link', ''))
            }
            metadatas.append(metadata)
            ids.append(f"news_{datetime.now().timestamp()}_{idx}")

        try:
            self.news_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} news items to vector database")
        except Exception as e:
            print(f"Error adding news data: {e}")

    def add_market_analysis(self, symbol: str, analysis_text: str, data_summary: Dict):
        """Add market analysis to vector database"""
        try:
            document = f"Market Analysis for {symbol}: {analysis_text}"
            metadata = {
                'symbol': symbol,
                'timestamp': str(datetime.now()),
                'type': 'market_analysis',
                'data_summary': json.dumps(data_summary, default=str)
            }

            doc_id = f"analysis_{symbol}_{datetime.now().timestamp()}"

            self.analysis_collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"Error adding market analysis: {e}")

    def search_news(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant news based on query"""
        try:
            results = self.news_collection.query(
                query_texts=[query],
                n_results=n_results
            )

            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })

            return formatted_results
        except Exception as e:
            print(f"Error searching news: {e}")
            return []

    def search_analysis(self, query: str, symbol: str = None, n_results: int = 3) -> List[Dict]:
        """Search for market analysis"""
        try:
            where_filter = {}
            if symbol:
                where_filter['symbol'] = symbol

            results = self.analysis_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter if where_filter else None
            )

            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })

            return formatted_results
        except Exception as e:
            print(f"Error searching analysis: {e}")
            return []

    def get_recent_news(self, limit: int = 10) -> List[Dict]:
        """Get recent news from database"""
        try:
            # ChromaDB doesn't support direct sorting, so we'll get more results and sort manually
            results = self.news_collection.get(limit=limit * 2)

            if not results['documents']:
                return []

            # Combine results with metadata
            news_items = []
            for i in range(len(results['documents'])):
                news_items.append({
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i] if i < len(results['metadatas']) else {}
                })

            # Sort by timestamp (if available)
            try:
                news_items.sort(
                    key=lambda x: datetime.fromisoformat(x['metadata'].get('timestamp', '1900-01-01')),
                    reverse=True
                )
            except:
                pass

            return news_items[:limit]
        except Exception as e:
            print(f"Error getting recent news: {e}")
            return []

    def clear_old_data(self, days_old: int = 30):
        """Clear data older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            # This would require custom implementation for ChromaDB
            # For now, we'll just print a message
            print(f"Note: Manual cleanup of data older than {days_old} days recommended")
        except Exception as e:
            print(f"Error clearing old data: {e}")

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            news_count = self.news_collection.count()
            analysis_count = self.analysis_collection.count()

            return {
                'news_items': news_count,
                'analysis_items': analysis_count,
                'total_items': news_count + analysis_count,
                'database_path': self.db_path
            }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {'error': str(e)}