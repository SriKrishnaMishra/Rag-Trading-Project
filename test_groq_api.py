#!/usr/bin/env python3
"""
Groq API Key Testing Tool (UPDATED FOR 2025)
This script tests if your Groq API key is valid and working
"""

import os
from dotenv import load_dotenv


def test_groq_api_key():
    """Test Groq API key functionality"""
    print("🔑 Testing Groq API Key...")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv('GROQ_API_KEY')

    if not api_key:
        print("❌ No GROQ_API_KEY found in .env file")
        print("\n📝 To fix this:")
        print("1. Create a .env file in your project folder")
        print("2. Add this line: GROQ_API_KEY=your_actual_api_key_here")
        print("3. Get a free API key at: https://console.groq.com")
        return False

    if api_key == 'your_groq_api_key_here':
        print("❌ Please replace 'your_groq_api_key_here' with your actual API key")
        return False

    print(f"✅ API Key found: {api_key[:20]}...{api_key[-10:]}")

    # Test if groq package is installed
    try:
        from groq import Groq
        print("✅ Groq package is installed")
    except ImportError:
        print("❌ Groq package not installed")
        print("💡 Install it with: pip install groq")
        return False

    # Test API connection with UPDATED MODEL
    try:
        print("🔍 Testing API connection...")
        client = Groq(api_key=api_key)

        # Simple test message with CURRENT SUPPORTED MODEL
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # ✅ UPDATED to supported model
            messages=[
                {"role": "user",
                 "content": "Hello! Just testing the API connection. Please respond with 'API test successful'."}
            ],
            max_tokens=50,
            temperature=0.1
        )

        if response.choices and response.choices[0].message:
            answer = response.choices[0].message.content
            print(f"✅ API Response: {answer}")
            print("🎉 Your Groq API key is working perfectly!")

            # Test financial query with UPDATED MODEL
            print("\n📊 Testing financial analysis capability...")
            financial_response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # ✅ UPDATED to supported model
                messages=[
                    {"role": "system",
                     "content": "You are a financial analyst. Provide brief, professional responses."},
                    {"role": "user", "content": "What are the key factors to consider when analyzing a stock?"}
                ],
                max_tokens=200,
                temperature=0.7
            )

            if financial_response.choices:
                financial_answer = financial_response.choices[0].message.content
                print(f"💼 Financial Analysis Test: {financial_answer[:200]}...")
                print("✅ Financial analysis capability confirmed!")

            return True
        else:
            print("❌ No response from API")
            return False

    except Exception as e:
        print(f"❌ API Error: {e}")

        # Check common error types
        error_str = str(e).lower()
        if "invalid api key" in error_str or "unauthorized" in error_str:
            print("🔑 Your API key appears to be invalid")
            print("💡 Please check:")
            print("   - API key is correct (no extra spaces)")
            print("   - API key is active at https://console.groq.com")
            print("   - You have API quota remaining")
        elif "rate limit" in error_str:
            print("⏰ Rate limit exceeded - your key works but you've hit the limit")
            print("💡 Wait a moment and try again")
        elif "network" in error_str or "connection" in error_str:
            print("🌐 Network connection issue - check your internet")
        elif "model" in error_str and "decommissioned" in error_str:
            print("🚫 Model has been decommissioned")
            print("💡 This script has been updated with current models")
        else:
            print("❓ Unknown error - your key might still be valid")

        return False


def get_available_models():
    """Get list of currently available Groq models"""
    print("🤖 Current Groq Models (as of 2025):")
    print("=" * 50)

    models = {
        "llama-3.1-70b-versatile": "Best overall performance, recommended",
        "llama-3.1-8b-instant": "Faster responses, good for most tasks",
        "llama-3.2-90b-vision-preview": "Supports images and text",
        "gemma2-9b-it": "Google's Gemma model, efficient",
        "mixtral-8x7b-32768": "If still available, high context window"
    }

    for model, description in models.items():
        print(f"✅ {model}")
        print(f"   📝 {description}\n")

    print("💡 For the most current list, visit: https://console.groq.com/docs/models")


def check_project_compatibility():
    """Check if the project will work with current setup"""
    print("\n🔧 Checking Project Compatibility...")
    print("=" * 50)

    # Check required packages
    required_packages = [
        ('streamlit', 'Web interface'),
        ('yfinance', 'Stock data'),
        ('pandas', 'Data processing'),
        ('plotly', 'Charts'),
        ('requests', 'Web requests'),
        ('groq', 'AI chatbot')
    ]

    missing_packages = []

    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (MISSING)")
            missing_packages.append(package)

    # Check optional packages
    print("\n📊 Optional Packages (for advanced features):")
    optional_packages = [
        ('chromadb', 'Vector database'),
        ('sentence_transformers', 'Text embeddings'),
        ('talib', 'Technical analysis'),
        ('beautifulsoup4', 'Web scraping')
    ]

    for package, description in optional_packages:
        try:
            if package == 'talib':
                import talib
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"⚠️  {package} - {description} (optional)")

    # Summary
    print(f"\n📋 Summary:")
    if not missing_packages:
        print("✅ All required packages installed!")
        print("🚀 Your setup is ready for the full project!")
        return True
    else:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False


def show_api_key_setup_guide():
    """Show step-by-step API key setup"""
    print("\n📚 API Key Setup Guide:")
    print("=" * 50)
    print("""
🔑 How to get a FREE Groq API key:

1. 🌐 Visit: https://console.groq.com
2. 📧 Sign up with your email (it's free!)
3. ✅ Verify your email address
4. 🔑 Go to API Keys section
5. ➕ Click "Create API Key"
6. 📋 Copy your API key

📝 How to add it to your project:

1. 📁 Create a file called '.env' in your project folder
2. ✏️  Add this line to the file:
   GROQ_API_KEY=paste_your_actual_key_here
3. 💾 Save the file
4. 🚀 Run this test again!

💡 Tips:
- Keep your API key secret (never share it)
- The free tier includes generous usage limits
- No credit card required for free tier
""")


def main():
    """Main test function"""
    print("🚀 Groq API Key & Project Compatibility Tester (2025 Edition)")
    print("=" * 60)

    # Show available models
    get_available_models()

    # Test API key
    api_working = test_groq_api_key()

    # Check project compatibility
    project_ready = check_project_compatibility()

    # Final assessment
    print("\n" + "=" * 60)
    print("🎯 FINAL ASSESSMENT:")

    if api_working and project_ready:
        print("🎉 PERFECT! Your setup is 100% ready!")
        print("🚀 You can run the trading app with: streamlit run main.py")
    elif api_working and not project_ready:
        print("🔧 Your API key works, but install missing packages first")
        print("💡 Run: python install_dependencies.py")
    elif not api_working and project_ready:
        print("📦 Packages ready, but fix your Groq API key")
        show_api_key_setup_guide()
    else:
        print("🛠️  Need to fix both API key and install packages")
        show_api_key_setup_guide()
        print("💡 Then run: python install_dependencies.py")

    print("\n🤔 Questions? Check the SETUP.md file for detailed instructions!")


if __name__ == "__main__":
    main()