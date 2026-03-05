import sys
import os

# Add data_analysis_agent dir to path so agent.py imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_analysis_agent'))

from data_analysis_agent.app import run as run_data_agent

run_data_agent()
