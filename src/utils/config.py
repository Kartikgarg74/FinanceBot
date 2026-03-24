"""Configuration loader with YAML parsing, env var substitution, and user overrides."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${ENV_VAR} patterns with actual environment variables."""
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")
        return pattern.sub(replacer, obj)
    elif isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigLoader:
    """Loads and merges YAML configuration with env var substitution."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"

        # Load .env file
        env_path = self.project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    def load(self, user: str = "kartik") -> dict:
        """Load merged configuration for a given user."""
        # Load base config
        default_path = self.config_dir / "default.yaml"
        if not default_path.exists():
            raise FileNotFoundError(f"Default config not found: {default_path}")

        with open(default_path) as f:
            config = yaml.safe_load(f) or {}

        # Load user preferences and merge
        user_prefs_path = self.config_dir / "users" / user / "preferences.yaml"
        if user_prefs_path.exists():
            with open(user_prefs_path) as f:
                user_prefs = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_prefs)

        # Substitute environment variables
        config = _substitute_env_vars(config)

        return config

    def load_user_profile(self, user: str = "kartik") -> dict:
        """Load user profile YAML as a raw dict."""
        profile_path = self.config_dir / "users" / user / "profile.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"User profile not found: {profile_path}")

        with open(profile_path) as f:
            return yaml.safe_load(f) or {}

    def load_finance_config(self, market: str) -> dict:
        """Load a finance market config (zerodha, alpaca, binance)."""
        config_path = self.config_dir / f"{market}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Finance config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        return _substitute_env_vars(config)

    def load_news_config(self) -> dict:
        """Load news & sentiment config."""
        config_path = self.config_dir / "news.yaml"
        if not config_path.exists():
            return {}

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        return _substitute_env_vars(config)
