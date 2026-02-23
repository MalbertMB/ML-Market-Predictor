"""
Project Name: ML-Predictor
File Name: config.py
Description:
    This module loads and manages configuration settings for the ML-Predictor project, including API keys and other constants.

Author: Albert Marín Blasco
"""


import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the root directory
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER_URL = os.getenv("ALPACA_PAPER_URL")

# Fail fast if the keys are missing
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the .env file.")