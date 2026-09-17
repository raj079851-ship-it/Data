# -*- coding: utf-8 -*-
"""
Unicode Safety and Sanitization Utilities for DataMind AI.

Prevents UnicodeEncodeError when passing strings containing UTF-16 surrogate pairs
or malformed surrogate code points (U+D800 to U+DFFF) to Streamlit components,
Tornado HTTP responses, or JSON serializers.
"""

def sanitize_unicode(text: str) -> str:
    """
    Sanitizes a string to ensure it is strictly valid UTF-8 encodable Unicode.
    
    1. Recombines valid UTF-16 surrogate pairs (e.g. \ud83d\udd0a -> \U0001F50A) into
       their proper 4-byte Unicode code points.
    2. Safely replaces or discards lone/orphaned surrogate code points (0xD800-0xDFFF)
       that are illegal in UTF-8.
    3. Preserves all valid emojis, multi-byte international characters, punctuation,
       and HTML/JS formatting intact.
    
    Args:
        text (str): Input text that may contain valid emojis or malformed surrogates.
        
    Returns:
        str: Guaranteed UTF-8 encodable Unicode string.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Step 1: Re-encode valid surrogate pairs using surrogatepass to heal pairs into single code points
    try:
        text = text.encode("utf-16", errors="surrogatepass").decode("utf-16")
    except Exception:
        pass

    # Step 2: Ensure the string can be encoded to UTF-8 without raising UnicodeEncodeError
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        # Replace remaining isolated/unpaired surrogate code points with Unicode replacement character
        return text.encode("utf-8", errors="replace").decode("utf-8")
