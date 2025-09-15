#!/usr/bin/env python3
"""
RAG Application Startup Script
Simple script to start the RAG application
"""

import sys
import subprocess
import os

def main():
    """Main startup function"""
    print("🚀 Private Local RAG — Autoimmune Liver (AIH/PBC/PSC)")
    print("=" * 40)
    print("Choose an option:")
    print("1. Start Web UI (Gradio)")
    print("2. Run Command Line Interface")
    print("3. Test with sample question")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🌐 Starting Gradio Web UI...")
            print("The web interface will open at: http://127.0.0.1:7860")
            subprocess.run([sys.executable, "gradio_app.py"])
            break
            
        elif choice == "2":
            print("\n💻 Starting Command Line Interface...")
            subprocess.run([sys.executable, "main.py"])
            break
            
        elif choice == "3":
            print("\n🧪 Testing with sample question...")
            subprocess.run([sys.executable, "main.py", "--ask_once", "Initial steroid regimen for AIH flare?"])
            break
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
