"""
Security & Credential Vault Module
Provides defense-in-depth credential management:
- Symmetric encryption (Fernet AES-128-CBC + HMAC-SHA256) for passwords, tokens, private keys
- CredentialSanitizer to mask sensitive values from logs, queries, and tracebacks
- Master Key derivation via PBKDF2-HMAC-SHA256 or environment injection
"""

import os
import re
import base64
import json
import logging
from typing import Dict, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger("DataMind.Security")


class CredentialSanitizer:
    """Masks secrets, passwords, connection strings, and tokens from logs and tracebacks."""

    PATTERNS = [
        # DB connection strings: postgres://user:pass@host
        (re.compile(r'([a-zA-Z0-9_+.-]+://[^:]+:)([^@\s]+)(@)', re.IGNORECASE), r'\1********\3'),
        # Key-value pairs: password=xxx, pwd=xxx, token=xxx, secret=xxx, api_key=xxx
        (re.compile(r'((?:password|pwd|secret|token|api[_-]?key|private[_-]?key|auth)\s*[:=]\s*["\']?)([^"\'\s,;}]+)(["\']?)', re.IGNORECASE), r'\1********\3'),
        # Bearer tokens
        (re.compile(r'(Bearer\s+)[a-zA-Z0-9_\-\.]{15,}', re.IGNORECASE), r'\1[REDACTED_BEARER_TOKEN]'),
        # RSA/EC/OPENSSH Private Keys
        (re.compile(r'-----BEGIN [A-Z ]+PRIVATE KEY-----[^-]+-----END [A-Z ]+PRIVATE KEY-----', re.DOTALL), r'[REDACTED_PRIVATE_KEY_BLOCK]'),
    ]

    @classmethod
    def sanitize(cls, text: Union[str, Exception, Any]) -> str:
        """Returns a sanitized string with all sensitive tokens and passwords masked."""
        if text is None:
            return ""
        s = str(text)
        for pattern, replacement in cls.PATTERNS:
            s = pattern.sub(replacement, s)
        return s

    @classmethod
    def mask_secret(cls, secret: Optional[str], keep_chars: int = 2) -> str:
        """Masks a secret string for UI preview (e.g. 'se********34')."""
        if not secret:
            return ""
        if len(secret) <= keep_chars * 2:
            return "********"
        return f"{secret[:keep_chars]}{'*' * (len(secret) - keep_chars * 2)}{secret[-keep_chars:]}"


class CredentialVault:
    """
    Encrypts and decrypts connector credentials at rest.
    Credentials remain encrypted in memory and configuration files;
    they are only decrypted transiently when establishing network sockets/sessions.
    """

    _DEFAULT_SALT = b"datamind_secure_analytics_salt_v1"

    def __init__(self, master_key: Optional[Union[str, bytes]] = None):
        self._fernet = self._init_cipher(master_key)

    def _init_cipher(self, master_key: Optional[Union[str, bytes]]) -> Fernet:
        # 1. Environment variable override
        env_key = os.environ.get("DATAMIND_ENCRYPTION_KEY")
        if env_key:
            try:
                # If already base64 32-bytes
                return Fernet(env_key.encode("utf-8") if isinstance(env_key, str) else env_key)
            except Exception:
                pass

        # 2. Provided master_key string/bytes
        raw_key = master_key or env_key or "DataMind-Enterprise-Secret-Key-2026"
        if isinstance(raw_key, str):
            raw_key = raw_key.encode("utf-8")

        # 3. Derive 32-byte Fernet key using PBKDF2-HMAC-SHA256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._DEFAULT_SALT,
            iterations=100_000,
        )
        derived = base64.urlsafe_b64encode(kdf.derive(raw_key))
        return Fernet(derived)

    def encrypt(self, plaintext: Union[str, dict, list]) -> str:
        """Encrypts string or serializable dict/list into an armored ciphertext token."""
        if plaintext is None:
            return ""
        if isinstance(plaintext, (dict, list)):
            data_bytes = json.dumps(plaintext).encode("utf-8")
        elif isinstance(plaintext, str):
            data_bytes = plaintext.encode("utf-8")
        else:
            data_bytes = str(plaintext).encode("utf-8")
        return self._fernet.encrypt(data_bytes).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypts an armored ciphertext token back into plaintext string."""
        if not ciphertext:
            return ""
        try:
            return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except Exception as e:
            logger.error(f"Credential decryption failure: {CredentialSanitizer.sanitize(str(e))}")
            raise ValueError("Failed to decrypt stored credentials (corrupted or wrong master key)")

    def decrypt_json(self, ciphertext: str) -> Any:
        """Decrypts ciphertext and parses as JSON object."""
        plain = self.decrypt(ciphertext)
        return json.loads(plain) if plain else None


# Global singleton instance for the platform runtime
default_vault = CredentialVault()
