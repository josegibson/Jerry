#!/usr/bin/env python3
"""
Validate Jerry installation and dependencies.

Run this script after cloning to check if everything is set up correctly.
"""
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.9+"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 9):
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (meets requirement >=3.9)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires >=3.9)")
        return False

def check_package_structure():
    """Check if package structure is correct"""
    print("\n🔍 Checking package structure...")
    required_files = [
        "pyproject.toml",
        "jerry/__init__.py",
        "jerry/cli.py",
        "jerry/__main__.py",
        "core/__init__.py",
        "core/agent_runtime.py",
    ]
    
    root = Path(__file__).parent
    all_good = True
    
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
            all_good = False
    
    return all_good

def check_pip():
    """Check if pip is available"""
    print("\n🔍 Checking pip...")
    try:
        import pip
        print(f"   ✅ pip is installed")
        return True
    except ImportError:
        print(f"   ❌ pip is not installed")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = {
        "setuptools": "required for installation",
        "wheel": "required for building",
        "typer": "core dependency",
        "rich": "core dependency",
        "dotenv": "core dependency (try 'python-dotenv')",
        "langchain_core": "optional: LLM support",
        "google.generativeai": "optional: Google Gemini",
        "openai": "optional: OpenAI",
        "chromadb": "optional: vector store",
    }
    
    installed = []
    missing = []
    
    for module, description in dependencies.items():
        try:
            __import__(module)
            installed.append(f"   ✅ {module} - {description}")
        except ImportError:
            missing.append(f"   ⚠️  {module} - {description}")
    
    for msg in installed:
        print(msg)
    
    if missing:
        print("\n   Missing optional dependencies:")
        for msg in missing:
            print(msg)
    
    return len(installed) > 0

def check_env_file():
    """Check if .env file exists"""
    print("\n🔍 Checking for .env file...")
    root = Path(__file__).parent
    env_file = root / ".env"
    
    if env_file.exists():
        print("   ✅ .env file found")
        # Check for API keys
        with open(env_file) as f:
            content = f.read()
            has_google = "GOOGLE_API_KEY" in content
            has_openai = "OPENAI_API_KEY" in content
            
            if has_google:
                print("   ✅ GOOGLE_API_KEY configured")
            if has_openai:
                print("   ✅ OPENAI_API_KEY configured")
            
            if not (has_google or has_openai):
                print("   ⚠️  No API keys found in .env file")
                print("      Add at least one: GOOGLE_API_KEY or OPENAI_API_KEY")
        return True
    else:
        print("   ⚠️  .env file not found")
        print("      Create one with your API keys:")
        print("      GOOGLE_API_KEY=your-key-here")
        print("      OPENAI_API_KEY=your-key-here")
        return False

def print_next_steps():
    """Print next steps for setup"""
    print("\n" + "="*60)
    print("📋 NEXT STEPS")
    print("="*60)
    print("""
1. Install Jerry and dependencies:
   
   pip install -e .

2. Create .env file with API keys:
   
   Create a .env file in this directory with:
   GOOGLE_API_KEY=your-google-api-key
   OPENAI_API_KEY=your-openai-api-key

3. Test the installation:
   
   jerry --help

4. Run Jerry in any directory:
   
   cd /path/to/your/project
   jerry

📚 Documentation:
   - Installation Guide: INSTALL.md
   - Quick Start: docs/QUICKSTART.md
   - Full Guide: README.md
""")

def main():
    """Run all validation checks"""
    print("="*60)
    print("🎩 Jerry Installation Validator")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_package_structure(),
        check_pip(),
        check_dependencies(),
        check_env_file(),
    ]
    
    print("\n" + "="*60)
    if all(checks[:3]):  # Core checks
        print("✅ Core requirements met! Ready to install.")
    else:
        print("❌ Some core requirements missing. Fix them before installing.")
    
    if not checks[3]:
        print("⚠️  Dependencies not installed yet - run installation command.")
    
    if not checks[4]:
        print("⚠️  API keys not configured - create .env file.")
    
    print_next_steps()

if __name__ == "__main__":
    main()
