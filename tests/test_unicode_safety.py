# -*- coding: utf-8 -*-
"""
Regression test suite for Unicode safety and surrogate character handling.
Ensures valid emojis and audio controls are preserved while malformed or
unpaired surrogate characters are safely handled without raising UnicodeEncodeError.
"""

import unittest
import json
import os
import sys

# Ensure modules directory is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.unicode_helper import sanitize_unicode


class TestUnicodeSafety(unittest.TestCase):
    """Test suite verifying Unicode safety, emoji preservation, and surrogate handling."""

    def test_valid_emojis_preserved(self):
        """Verify that standard multi-byte emojis and symbols are completely preserved."""
        emojis = "🔊 ⏹️ 📊 🚀 🤖 💡 🛡️ 📈 🗄️ 🧑‍🤝‍🧑 📦 🧠 🛒"
        sanitized = sanitize_unicode(emojis)
        self.assertEqual(sanitized, emojis)
        
        # Verify UTF-8 encoding succeeds without error
        encoded = sanitized.encode("utf-8")
        self.assertTrue(len(encoded) > 0)

    def test_surrogate_pair_healing(self):
        """Verify that valid UTF-16 surrogate pairs are healed into genuine Unicode characters."""
        # 0xD83D and 0xDD0A is the UTF-16 surrogate pair for 🔊 (U+1F50A)
        surrogate_speaker = "Volume: " + chr(0xD83D) + chr(0xDD0A)
        sanitized = sanitize_unicode(surrogate_speaker)
        self.assertIn("🔊", sanitized)
        self.assertNotIn(chr(0xD83D), sanitized)
        
        # Verify UTF-8 encoding succeeds without error
        encoded = sanitized.encode("utf-8")
        self.assertIsInstance(encoded, bytes)

    def test_malformed_isolated_surrogates(self):
        """Verify that lone or malformed surrogate characters are safely handled without crash."""
        # Isolated high surrogate
        lone_high = "Broken streaming text " + chr(0xD83D) + " with trailing content"
        clean_high = sanitize_unicode(lone_high)
        # Must encode to UTF-8 without raising UnicodeEncodeError
        encoded_high = clean_high.encode("utf-8")
        self.assertTrue(len(encoded_high) > 0)

        # Isolated low surrogate
        lone_low = "Malformed response " + chr(0xDD0A) + " here"
        clean_low = sanitize_unicode(lone_low)
        encoded_low = clean_low.encode("utf-8")
        self.assertTrue(len(encoded_low) > 0)

        # Mixed valid emoji and lone surrogate
        mixed = "Valid 🚀 and broken " + chr(0xD83D) + " and another valid 🤖"
        clean_mixed = sanitize_unicode(mixed)
        self.assertIn("🚀", clean_mixed)
        self.assertIn("🤖", clean_mixed)
        encoded_mixed = clean_mixed.encode("utf-8")
        self.assertTrue(len(encoded_mixed) > 0)

    def test_voice_player_template_clean(self):
        """Verify that app.py contains no literal surrogate escape pairs in voice_player_html."""
        app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            app_code = f.read()

        # Ensure the offending surrogate escape is absent
        self.assertNotIn(r"\uD83D\uDD0A", app_code, "Found raw surrogate pair escape in app.py!")
        # Ensure the proper code point or emoji is used
        self.assertTrue(
            r"\U0001F50A" in app_code or "🔊" in app_code,
            "Proper speaker emoji or \U0001F50A not found in app.py!"
        )

    def test_simulated_components_html_rendering(self):
        """Simulate Streamlit components.html encoding with assistant response containing emojis and surrogates."""
        # Simulated assistant chat responses
        simulated_responses = [
            "Great news! Revenue increased by 14.2% 📈 across all segments 🚀.",
            "Alert: High churn detected in Month-to-month contracts ⚠️ 🔴.",
            "Simulating broken API stream: " + chr(0xD83D) + chr(0xDD0A) + " audio response ready.",
            "Malformed external chunk: " + chr(0xD83D) + " partial text without low surrogate."
        ]

        # Extract voice player template from app.py
        app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            app_code = f.read()

        start_marker = 'voice_player_html = """'
        start = app_code.find(start_marker) + len(start_marker)
        end_marker = '""".replace("__RAW_TEXT__", escaped_text)'
        end = app_code.find(end_marker, start)
        raw_template = app_code[start:end]

        for resp in simulated_responses:
            clean_resp = sanitize_unicode(resp)
            escaped = json.dumps(clean_resp, ensure_ascii=False)
            
            html = raw_template.replace("__RAW_TEXT__", escaped)
            html = html.replace("__LANG_CODE__", "en-US")
            html = html.replace("__GENDER_VAL__", "female")
            html = html.replace("__GENDER_LABEL__", "Female")
            html = html.replace("__SPEED_VAL__", "1.0")
            html = html.replace("__SHOULD_AUTO__", "false")
            
            safe_html = sanitize_unicode(html)
            
            # This simulates exactly what Streamlit components.html does internally
            try:
                utf8_bytes = safe_html.encode("utf-8")
                self.assertTrue(len(utf8_bytes) > 0)
            except UnicodeEncodeError as e:
                self.fail(f"UnicodeEncodeError raised on simulated components.html rendering: {e}")

    def test_non_string_inputs(self):
        """Verify that None, numbers, booleans, and dictionaries are handled gracefully."""
        self.assertEqual(sanitize_unicode(None), "")
        self.assertEqual(sanitize_unicode(12345), "12345")
        self.assertEqual(sanitize_unicode(True), "True")


if __name__ == "__main__":
    unittest.main()
