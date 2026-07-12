"""Make il_rag importable when pytest is launched from anywhere.

All tests are offline: every chat()/embed() call and the Chroma retriever are
monkeypatched, so no API key and no index are needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
