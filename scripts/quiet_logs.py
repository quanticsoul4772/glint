"""Imported first by tools_ab: loguru's default handler level, set before τ² is imported.

τ²'s registry logs about 25 lines (DEBUG, with its whole registry as JSON) when it is imported, before any handler
Glint installs, at the top of every run log. LOGURU_LEVEL is read once, when loguru is first imported, so it is set
here, before any import that loads loguru. Each simulation's own task.log is added with an explicit DEBUG level and
keeps every record. An explicit LOGURU_LEVEL in the environment wins.
"""
import os

os.environ.setdefault("LOGURU_LEVEL", "INFO")
