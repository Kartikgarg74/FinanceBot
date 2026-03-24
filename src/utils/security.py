"""Security utilities - prompt sanitization, encryption, path validation, LLM guardrails."""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Prompt Injection Protection ---

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?above",
    r"system\s*:\s*",
    r"<\|?system\|?>",
    r"<\|?assistant\|?>",
    r"<\|?user\|?>",
    r"```\s*system",
    r"IMPORTANT:\s*ignore",
    r"override\s+instructions",
    r"forget\s+(all\s+)?prior",
    r"new\s+instructions?\s*:",
    r"you\s+are\s+now\s+a",
    r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+if",
    r"reveal\s+(your|the)\s+(system|api|secret|key|password)",
    r"output\s+(your|the)\s+prompt",
    r"what\s+is\s+your\s+system\s+prompt",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_prompt_input(text: str, max_length: int = 5000) -> str:
    """Sanitize user-controlled text before inserting into LLM prompts.

    - Truncates to max_length
    - Strips prompt injection patterns
    - Removes control characters
    """
    if not text:
        return ""

    # Truncate first
    text = text[:max_length]

    # Remove control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Defang injection attempts
    text = _INJECTION_RE.sub("[filtered]", text)

    return text


def detect_prompt_injection(text: str) -> bool:
    """Check if text contains prompt injection patterns. Returns True if suspicious."""
    if not text:
        return False
    return bool(_INJECTION_RE.search(text))


def sanitize_for_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text for safe use in filenames. Prevents path traversal."""
    if not text:
        return "unknown"
    text = re.sub(r"[/\\:*?\"<>|.\x00]", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = text.strip().replace(" ", "_")
    return text[:max_length] if text else "unknown"


# --- Path Traversal Protection ---

def validate_safe_path(base_dir: Path, user_input: str) -> Path:
    """Validate that a user-supplied path component doesn't escape base_dir."""
    if ".." in user_input or "/" in user_input or "\\" in user_input:
        raise ValueError(f"Invalid path component: {user_input}")

    safe = re.sub(r"[^\w-]", "", user_input)
    if not safe:
        raise ValueError(f"Path component is empty after sanitization: {user_input}")

    resolved = (base_dir / safe).resolve()
    base_resolved = base_dir.resolve()

    if not str(resolved).startswith(str(base_resolved)):
        raise ValueError(f"Path traversal detected: {user_input}")

    return resolved


# --- Authenticated Encryption (Fernet) ---

def _get_encryption_key() -> bytes:
    """Get Fernet encryption key from env or derive from machine-specific data."""
    key = os.environ.get("KARTIKAI_COOKIE_KEY", "")
    if key:
        # Fernet key must be 32 url-safe base64-encoded bytes
        return base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest())

    # Derive from machine-specific data
    import socket
    import getpass
    machine_id = f"{socket.gethostname()}:{getpass.getuser()}:kartikai_v2"
    return base64.urlsafe_b64encode(hashlib.sha256(machine_id.encode()).digest())


def encrypt_data(data: bytes) -> bytes:
    """Encrypt data using Fernet (AES-128-CBC with HMAC authentication).

    Falls back to base64 encoding if cryptography library is not installed.
    """
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_get_encryption_key())
        return f.encrypt(data)
    except ImportError:
        logger.warning("cryptography library not installed — using base64 encoding (NOT secure)")
        return base64.urlsafe_b64encode(data)


def decrypt_data(data: bytes) -> bytes:
    """Decrypt data encrypted with encrypt_data."""
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_get_encryption_key())
        return f.decrypt(data)
    except ImportError:
        return base64.urlsafe_b64decode(data)
    except Exception as e:
        logger.error("Decryption failed: %s", sanitize_error(e))
        raise


# --- Secret Storage (for TOTP, API keys at rest) ---

def encrypt_secret(secret: str) -> str:
    """Encrypt a secret string for storage. Returns base64-encoded ciphertext."""
    return encrypt_data(secret.encode()).decode("utf-8")


def decrypt_secret(encrypted: str) -> str:
    """Decrypt a secret string from storage."""
    return decrypt_data(encrypted.encode()).decode("utf-8")


# --- HMAC Signing for Telegram Callbacks ---

_cached_hmac_key: bytes | None = None


def _get_hmac_key() -> bytes:
    """Get HMAC key for signing Telegram callback data. Cached per process."""
    global _cached_hmac_key
    if _cached_hmac_key is not None:
        return _cached_hmac_key

    key = os.environ.get("KARTIKAI_HMAC_KEY", "")
    if key:
        _cached_hmac_key = hashlib.sha256(key.encode()).digest()
        return _cached_hmac_key

    # Derive from Telegram bot token (better than hardcoded)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if token:
        _cached_hmac_key = hashlib.sha256(f"hmac:{token}:kartikai".encode()).digest()
        return _cached_hmac_key

    # Generate ephemeral key (cached for session lifetime)
    logger.warning("No HMAC key configured — using ephemeral key (callbacks won't survive restart)")
    _cached_hmac_key = hashlib.sha256(secrets.token_bytes(32)).digest()
    return _cached_hmac_key


def sign_callback_data(data: str) -> str:
    """Sign callback data with HMAC. Returns 'data|signature'."""
    key = _get_hmac_key()
    sig = hmac.new(key, data.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{data}|{sig}"


def verify_callback_data(signed_data: str) -> str | None:
    """Verify and extract callback data. Returns data if valid, None if tampered."""
    if "|" not in signed_data:
        return None
    data, sig = signed_data.rsplit("|", 1)
    key = _get_hmac_key()
    expected = hmac.new(key, data.encode(), hashlib.sha256).hexdigest()[:16]
    if hmac.compare_digest(sig, expected):
        return data
    return None


# --- File Permission Hardening ---

def secure_directory(path: Path) -> None:
    """Set directory permissions to owner-only (0700)."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, stat.S_IRWXU)


def secure_file(path: Path) -> None:
    """Set file permissions to owner-only (0600)."""
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


# --- Error Sanitization ---

def sanitize_error(error: Exception, max_length: int = 200) -> str:
    """Sanitize an exception message for safe logging.

    Removes file paths, API keys, tokens, and other sensitive data.
    """
    msg = str(error)[:max_length]
    # API keys
    msg = re.sub(r"sk-[a-zA-Z0-9-]{20,}", "[api_key]", msg)
    msg = re.sub(r"gsk_[a-zA-Z0-9]{20,}", "[api_key]", msg)
    # Hex hashes / tokens
    msg = re.sub(r"[a-f0-9]{32,}", "[hash]", msg)
    # File paths
    msg = re.sub(r"/[\w/.-]+\.\w+", "[path]", msg)
    # Passwords
    msg = re.sub(r"password[=:]\S+", "password=[redacted]", msg, flags=re.IGNORECASE)
    # Bearer tokens
    msg = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", msg)
    return msg


# --- LLM Output Validation ---

def safe_parse_json(text: str, fallback: dict | None = None) -> dict | None:
    """Safely parse JSON from LLM output with multiple extraction strategies.

    Returns the parsed dict, or fallback if all parsing fails.
    """
    if not text:
        return fallback

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    # Strategy 3: Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("All JSON parsing strategies failed for LLM output: %s", text[:100])
    return fallback


def validate_llm_score(value, min_val: float = -5, max_val: float = 5, default: float = 0) -> float:
    """Validate and clamp a numeric score from LLM output."""
    try:
        v = float(value)
        return max(min_val, min(v, max_val))
    except (TypeError, ValueError):
        return default


def validate_llm_confidence(value, default: float = 50) -> float:
    """Validate confidence score (0-100) from LLM output."""
    return validate_llm_score(value, 0, 100, default)


def validate_trade_amount(
    quantity: int | float,
    price: float,
    max_capital: float,
    max_position_pct: float = 10.0,
) -> int | float:
    """Sanity-check a trade amount. Returns 0 if suspicious."""
    if quantity <= 0 or price <= 0:
        return 0
    total_value = quantity * price
    max_allowed = max_capital * (max_position_pct / 100)
    if total_value > max_allowed:
        logger.warning(
            "Trade amount %.2f exceeds max %.2f (%.1f%% of capital) — rejecting",
            total_value, max_allowed, max_position_pct,
        )
        return 0
    return quantity
