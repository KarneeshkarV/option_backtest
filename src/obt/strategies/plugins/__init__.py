"""Plugin directory for strategies.

Each ``*.py`` file registers a strategy via a top-level ``@strategy("name")``
decorator and is imported from ``obt.strategies.spec.discover_plugins``.
"""
