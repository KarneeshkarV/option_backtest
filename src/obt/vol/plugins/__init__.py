"""Plugin directory for volatility models.

Each ``*.py`` file registers a model via a top-level ``@vol_model("name")``
decorator and is imported from ``obt.vol.spec.discover_plugins``.
"""
