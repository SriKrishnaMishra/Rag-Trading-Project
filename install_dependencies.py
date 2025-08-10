#!/usr/bin/env python3
"""
Smart dependency installer for AI Trading Analysis Hub
This script installs dependencies with fallback options for problematic packages
"""

import subprocess
import sys
import platform
import os


def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def install_basic_requirements():
    """Install basic requirements that usually work without issues"""
    basic_packages = [
        "streamlit>=1.28.0",
        "yfinance>=0.2.20",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "feedparser>=3.0.0",
        "python-dotenv>=1.0.0"
    ]

    print("📦 Installing basic packages...")
    for package in basic_packages:
        success = run_command(f'pip install "{package}"', f"Installing {package.split('>=')[0]}")
        if not success:
            print(f"⚠️  Failed to install {package}, but continuing...")


def install_ai_packages():
    """Install AI-related packages"""
    print("\n🤖 Installing AI packages...")

    # Try Groq
    if not run_command('pip install groq>=0.4.0', "Installing Groq API client"):
        print("⚠️  Groq installation failed - chatbot features may not work")

    # Try sentence-transformers (might take time)
    print("📊 Installing sentence-transformers (this may take a few minutes)...")
    if not run_command('pip install sentence-transformers>=2.2.0', "Installing sentence transformers"):
        print("⚠️  Sentence transformers failed - using fallback embeddings")

    # Try ChromaDB
    if not run_command('pip install chromadb>=0.4.0', "Installing ChromaDB"):
        print("⚠️  ChromaDB installation failed - using memory storage")


def install_talib_alternatives():
    """Install TA-Lib or alternatives"""
    print("\n📈 Installing technical analysis libraries...")

    system = platform.system().lower()

    # First try regular TA-Lib installation
    if run_command('pip install TA-Lib', "Installing TA-Lib"):
        print("✅ TA-Lib installed successfully")
        return True

    print("⚠️  TA-Lib installation failed, trying alternatives...")

    # Try different TA-Lib installation methods
    if system == "windows":
        print("🪟 Detected Windows - trying Windows-specific TA-Lib installation")

        # Try downloading precompiled wheel
        arch = "win_amd64" if platform.architecture()[0] == "64bit" else "win32"
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

        talib_url = f"https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.19-{python_version}-{python_version}-{arch}.whl"

        if run_command(f'pip install {talib_url}', "Installing TA-Lib from precompiled wheel"):
            return True

    elif system == "darwin":  # macOS
        print("🍎 Detected macOS - trying Homebrew installation")
        run_command('brew install ta-lib', "Installing TA-Lib via Homebrew")
        run_command('pip install TA-Lib', "Installing TA-Lib Python wrapper")

    elif system == "linux":
        print("🐧 Detected Linux - installing TA-Lib dependencies")
        run_command('sudo apt-get update && sudo apt-get install -y build-essential', "Installing build tools")
        run_command(
            'wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib && ./configure --prefix=/usr && make && sudo make install',
            "Installing TA-Lib from source")
        run_command('pip install TA-Lib', "Installing TA-Lib Python wrapper")

    # Final attempt with alternative packages
    print("🔄 Trying alternative technical analysis packages...")
    alternatives = [
        "pandas-ta",
        "finta",
        "stockstats"
    ]

    for alt in alternatives:
        if run_command(f'pip install {alt}', f"Installing {alt} as TA-Lib alternative"):
            print(f"✅ Installed {alt} as alternative")
            return True

    print("⚠️  All TA-Lib alternatives failed - using manual calculations")
    return False


def create_requirements_fallback():
    """Create a fallback requirements.txt without problematic packages"""
    fallback_requirements = """# Core requirements that should work on all systems
streamlit>=1.28.0
yfinance>=0.2.20
requests>=2.31.0
beautifulsoup4>=4.12.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
feedparser>=3.0.0
python-dotenv>=1.0.0

# Try to install these, but app will work without them
# Uncomment if installation succeeds:
# groq>=0.4.0
# chromadb>=0.4.0
# sentence-transformers>=2.2.0
# TA-Lib>=0.4.0
"""

    with open('requirements_fallback.txt', 'w') as f:
        f.write(fallback_requirements)

    print("📝 Created requirements_fallback.txt with minimal dependencies")


def test_imports():
    """Test if critical packages can be imported"""
    print("\n🧪 Testing package imports...")

    test_packages = [
        ('streamlit', 'Streamlit web framework'),
        ('yfinance', 'Yahoo Finance data'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('plotly', 'Interactive charts'),
        ('requests', 'HTTP requests'),
        ('bs4', 'Web scraping'),
        ('groq', 'AI chatbot'),
        ('chromadb', 'Vector database'),
        ('sentence_transformers', 'Text embeddings')
    ]

    working_packages = []
    failed_packages = []

    for package, description in test_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
            working_packages.append(package)
        except ImportError:
            print(f"❌ {package} - {description} (not available)")
            failed_packages.append(package)

    print(f"\n📊 Results: {len(working_packages)} working, {len(failed_packages)} failed")

    if 'streamlit' in working_packages and 'yfinance' in working_packages:
        print("✅ Core functionality available - app should work!")
    else:
        print("❌ Critical packages missing - app may not work properly")

    return working_packages, failed_packages


def main():
    """Main installation process"""
    print("🚀 AI Trading Analysis Hub - Smart Installer")
    print("=" * 60)

    print(f"🐍 Python version: {sys.version}")
    print(f"💻 Platform: {platform.system()} {platform.architecture()[0]}")

    # Upgrade pip first
    print("\n⬆️  Upgrading pip...")
    run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip")

    # Install basic requirements
    install_basic_requirements()

    # Install AI packages
    install_ai_packages()

    # Try to install TA-Lib
    install_talib_alternatives()

    # Create fallback requirements
    create_requirements_fallback()

    # Test imports
    working, failed = test_imports()

    print("\n" + "=" * 60)
    print("🎉 Installation completed!")

    if len(failed) == 0:
        print("✅ All packages installed successfully!")
    elif 'streamlit' in working and 'yfinance' in working:
        print("✅ Core packages working - app will run with basic features")
        print("💡 Some advanced features may be limited due to missing packages")
    else:
        print("⚠️  Some critical packages failed to install")
        print("🔧 Try running: pip install -r requirements_fallback.txt")

    print("\n🚀 To start the app, run: streamlit run main.py")
    print("🔑 Don't forget to set up your .env file with API keys!")


if __name__ == "__main__":
    main()