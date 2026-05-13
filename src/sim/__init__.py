"""Metaverse server simulator (Flask) + synthetic event generator.

This package provides a minimal simulated environment so you can:
- generate "normal" metaverse telemetry events,
- trigger simulated attacks (DDoS, phishing),
- poll recent events from an HTTP endpoint,
- use the events as training/testing data for ML models.

Notes:
- This is an educational simulator (not a real metaverse server).
- We focus on producing telemetry signals that resemble what monitoring systems see.
"""
