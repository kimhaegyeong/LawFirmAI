# -*- coding: utf-8 -*-
"""
Google Gemini moderation adapter (optional)

Uses the google-generativeai SDK to run a lightweight moderation gate
before the internal rule-based pipeline. It is safe to import without
the SDK installed – when unavailable or when API key is missing, the
moderator becomes a no-op that always returns allowed.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GeminiModerator:
    """Optional Gemini moderation wrapper.

    When GOOGLE_API_KEY is not set or SDK is missing, acts as a no-op.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        self.model_name = model_name
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.enabled = bool(self.api_key) and os.getenv("GEMINI_MODERATION_ENABLED", "0") in ("1", "true", "True")
        self._client_ok = False
        self._model = None

        if not self.enabled:
            logger.info("Gemini moderation disabled or API key missing.")
            return

        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._client_ok = True
            logger.info(f"Gemini moderation initialized with model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Gemini SDK unavailable or init failed: {e}")
            self.enabled = False

    def moderate(self, text: str) -> Dict[str, Any]:
        """Run moderation. Returns a dict with fields:
        - allowed: bool
        - score: float (0~1: 1 means highly risky)
        - reasoning: str
        - details: dict
        """
        if not (self.enabled and self._client_ok and self._model):
            return {"allowed": True, "score": 0.0, "reasoning": "Gemini moderation disabled", "details": {}}

        try:
            # Prompt geared for safety classification without generating content
            prompt = (
                "You are a content safety classifier. Only return a one-line JSON with keys: "
                "allowed (true/false), risk (0..1), category (string). "
                "Classify the following user text for illegal activity assistance risk.\n\n"
                f"TEXT:\n{text}\n"
            )

            resp = self._model.generate_content(prompt)
            out = resp.text or ""
            allowed = True
            risk = 0.0
            category = "unknown"
            # Best-effort parse
            try:
                import json
                j = json.loads(out.strip())
                allowed = bool(j.get("allowed", True))
                risk = float(j.get("risk", 0.0))
                category = str(j.get("category", "unknown"))
            except Exception:
                # Fallback: if model outputs any safety blocks, consider medium risk
                allowed = True
                risk = 0.3
                category = "parsed_fallback"

            return {
                "allowed": allowed,
                "score": max(0.0, min(1.0, risk)),
                "reasoning": f"Gemini moderation category={category}",
                "details": {"category": category, "raw": out[:500]}
            }
        except Exception as e:
            logger.warning(f"Gemini moderation error: {e}")
            return {"allowed": True, "score": 0.0, "reasoning": f"Gemini error: {e}", "details": {}}


