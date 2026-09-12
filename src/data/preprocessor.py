"""
Text preprocessor and entity extractor for customer tweets.
"""

import re
from typing import Dict, Any

class TweetPreprocessor:
    DEVICE_PATTERNS = [
        (r"\biphone\s*(x|1[0-5]|[6-8](?:s)?(?:\s*plus)?)\b", "iPhone"),
        (r"\bipad\s*(pro|air|mini)?\b", "iPad"),
        (r"\bapple\s*watch\s*(series\s*[0-9])?\b", "Apple Watch"),
        (r"\bmacbook\s*(pro|air)?\b", "MacBook"),
        (r"\bairpods?\b", "AirPods"),
        (r"\bbeats\s*(x|studio|solo)?\b", "Beats")
    ]

    OS_PATTERNS = [
        (r"\bios\s*([0-9]+(?:\.[0-9]+)*)\b", "iOS"),
        (r"\bwatchos\s*([0-9]+(?:\.[0-9]+)*)\b", "watchOS"),
        (r"\bmacos\s*([a-zA-Z0-9]+(?:\.[0-9]+)*)\b", "macOS")
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        if not text:
            return ""
        # Remove mentions
        text = re.sub(r"@\w+", "", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, Any]:
        lower = text.lower()
        devices = []
        for pattern, brand in cls.DEVICE_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                devices.append(match.group(0).strip())

        os_versions = []
        for pattern, os_name in cls.OS_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                os_versions.append(match.group(0).strip())

        return {
            "devices": list(set(devices)),
            "os_versions": list(set(os_versions)),
            "has_device_info": len(devices) > 0,
            "has_os_info": len(os_versions) > 0
        }

    @classmethod
    def preprocess(cls, text: str) -> Dict[str, Any]:
        cleaned = cls.clean_text(text)
        entities = cls.extract_entities(text)
        return {
            "raw_text": text,
            "cleaned_text": cleaned,
            "entities": entities
        }
