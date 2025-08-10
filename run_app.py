#!/usr/bin/env python3
"""
Easy launcher script for AI Trading Analysis Hub
This script checks dependencies and launches the application
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'yfinance',
        'requests',
        'beautifulsoup4',
        'pandas',
        'numpy',
        'plotly',
        'groq',
        'chromadb',
        'sentence-transformers',
        'feedparser',
        'python-dotenv',
        'ta'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)

    return missing_packages


def install_requirements():
    """Install missing requirements"""
    print("\n🔧 Installing missing packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                       check=True, capture_output=True, text=True)
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("Please run manually: pip install -r requirements.txt")
        return False


def check_env_file():
    """Check if .env file exists and has required keys"""
    env_path = Path('.env')

    if not env_path.exists():
        print("❌ .env file not found")
        print("📋 Please copy .env.example to .env and add your API keys")
        return False

    # Read .env file
    env_content = env_path.read_text()

    if 'GROQ_API_KEY=' in env_content:
        groq_key_line = [line for line in env_content.split('\n') if 'GROQ_API_KEY=' in line][0]
        if 'your_groq_api_key_here' in groq_key_line or groq_key_line.split('=')[1].strip() == '':
            print("❌ GROQ_API_KEY not configured in .env file")
            print("🔑 Please add your Groq API key to .env file")
            print("Get free key at: https://console.groq.com")
            return False
        else:
            print("✅ GROQ_API_KEY configured")
            return True
    else:
        print("❌ GROQ_API_KEY not found in .env file")
        return False


def check_project_files():
    """Check if all project files exist"""
    required_files = [
        'main.py',
        'config.py',
        'data_fetcher.py',
        'news_scraper.py',
        'vector_database.py',
        'groq_chatbot.py',
        'visualization.py',
        'requirements.txt'
    ]

    missing_files = []

    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✅ {file}")

    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False

    return True


def create_sample_env():
    """Create sample .env file"""
    sample_env = """# API Keys for Financial Trading Analysis Hub
# Get your free Groq API key at: https://console.groq.com

GROQ_API_KEY=your_groq_api_key_here

# Optional API keys
NEWS_API_KEY=
ALPHA_VANTAGE_KEY=
"""

    with open('.env', 'w') as f:
        f.write(sample_env)

    print("✅ Created .env template file")
    print("🔑 Please add your Groq API key to .env file")


def launch_app():
    """Launch the Streamlit application"""
    print("\n🚀 Launching AI Trading Analysis Hub...")
    print("📱 Opening in your default browser...")
    print("🔗 URL: http://localhost:8501")
    print("\n⚠️  To stop the application, press Ctrl+C in this terminal")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")


def main():
    """Main launcher function"""
    print("🚀 AI Trading Analysis Hub - Setup & Launch")
    print("=" * 50)

    # Check Python version
    print("\n1️⃣ Checking Python version...")
    if not check_python_version():
        return

    # Check project files
    print("\n2️⃣ Checking project files...")
    if not check_project_files():
        print("\n❌ Some project files are missing. Please ensure all files are in the same directory.")
        return

    # Check requirements
    print("\n3️⃣ Checking dependencies...")
    missing_packages = check_requirements()

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        install = input("Install missing packages automatically? (y/n): ").lower().strip()

        if install == 'y':
            if not install_requirements():
                return
        else:
            print("Please install missing packages manually: pip install -r requirements.txt")
            return
    else:
        print("✅ All dependencies are installed")

    # Check .env file
    print("\n4️⃣ Checking configuration...")
    if not check_env_file():
        if not Path('.env').exists():
            create_env = input("Create .env template file? (y/n): ").lower().strip()
            if create_env == 'y':
                create_sample_env()
        return

    # All checks passed
    print("\n✅ All checks passed! Ready to launch.")
    print("\n5️⃣ Starting application...")

    launch_input = input("Launch the application now? (y/n): ").lower().strip()
    if launch_input == 'y':
        launch_app()
    else:
        print("👋 Run this script again when ready to launch")
        print("💡 Or run manually: streamlit run main.py")


if __name__ == "__main__":
    main()