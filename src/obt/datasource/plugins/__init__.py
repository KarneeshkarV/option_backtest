"""Plugin directory for spot data sources.

Each ``*.py`` file registers a source via a top-level ``@data_source("name")``
decorator. Add a source by dropping a module here and adding it to
``obt.datasource.spec.discover_plugins``.
"""
