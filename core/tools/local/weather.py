from __future__ import annotations

import os
import re
from typing import Any

import httpx

from . import Skill, SkillExecutionError

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Heavy freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_LOCATION_RE = re.compile(
    r"\b(?:weather|forecast|temperature|humidity|wind)\b(?:\s+\w+){0,5}?\s+"
    r"(?:in|for|at|near|around)\s+([A-Za-z][A-Za-z\s,.\-]{1,60}?)(?:\?|$|\.)",
    re.IGNORECASE,
)
_LOCATION_TRAILING_RE = re.compile(
    r"\s+(?:today|tonight|tomorrow|right\s+now|now|currently|this\s+\w+)\s*$",
    re.IGNORECASE,
)
_WEATHER_KEYWORDS = (
    "weather",
    "forecast",
    "temperature",
    "humidity",
    "wind",
    "rain",
    "snow",
    "sunny",
    "storm",
    "umbrella",
)


def _wmo_condition(code: int) -> str:
    return _WMO_CODES.get(code, f"Unknown condition ({code})")


def _extract_location_regex(transcript: str) -> str | None:
    match = _LOCATION_RE.search(transcript.strip())
    if not match:
        return None
    location = match.group(1).strip().rstrip(",. ")
    location = _LOCATION_TRAILING_RE.sub("", location).strip().rstrip(",. ")
    return location or None


class WeatherSkill(Skill):
    """Fast-path weather skill backed by Open-Meteo."""

    def __init__(self) -> None:
        self._units = os.getenv("WEATHER_UNITS", "celsius").strip().lower()
        self._timeout_s = float(os.getenv("WEATHER_TIMEOUT_MS", "5000")) / 1000.0
        self._spacy_nlp: Any | None = None
        self._spacy_loaded = False

    @property
    def name(self) -> str:
        return "weather"

    @property
    def exemplars(self) -> list[str]:
        return [
            "what's the weather in london",
            "will it rain tomorrow in berlin",
            "what is the forecast for paris today",
            "temperature in tokyo right now",
            "do i need an umbrella in seattle",
            "how windy is it in amsterdam",
            "weather around madrid",
            "humidity in lisbon today",
            "is it sunny in rome",
            "weather in new york tonight",
        ]

    @property
    def format_prompt(self) -> str:
        return (
            "You are Cass. Convert this structured weather tool result into a concise, "
            "natural spoken response. Keep it under two sentences. Mention location, "
            "condition, and temperature. Include wind or humidity only if notable."
        )

    def score(self, transcript: str) -> float:
        text = transcript.strip().lower()
        words = set(text.split())
        if not words:
            return 0.0
        hit = sum(1 for kw in _WEATHER_KEYWORDS if kw in text)
        if hit == 0:
            return 0.0
        # Scale: 1 hit → 0.55, 2 hits → 0.75, 3+ → 0.95.
        # Caps at 0.95 so a weak keyword hit never overrides a strong
        # embedding match for a *different* skill.
        if hit >= 3:
            return 0.95
        if hit >= 2:
            return 0.75
        return 0.55

    def can_handle(self, transcript: str) -> bool:
        text = transcript.strip().lower()
        return any(keyword in text for keyword in _WEATHER_KEYWORDS)

    def entities(self, transcript: str) -> dict[str, Any]:
        location = _extract_location_regex(transcript)
        if location:
            return {"location": location}

        # Optional spaCy GPE fallback when users omit explicit prepositions.
        if not self._spacy_loaded:
            self._spacy_loaded = True
            try:
                import spacy

                self._spacy_nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._spacy_nlp = None

        if self._spacy_nlp is not None:
            doc = self._spacy_nlp(transcript)
            for ent in doc.ents:
                if ent.label_ in {"GPE", "LOC", "FAC"}:
                    return {"location": ent.text.strip()}

        return {"location": None}

    async def execute(self, entities: dict[str, Any]) -> str:
        location = (entities.get("location") or "").strip() if entities else ""
        if not location:
            return (
                "status=missing_location; "
                "message=Please tell me the city, like 'weather in London'."
            )

        try:
            lat, lon, display_name = await self._geocode(location)
            current = await self._fetch_current(lat, lon)
        except ValueError as exc:
            raise SkillExecutionError(str(exc)) from exc
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise SkillExecutionError(
                "I can't reach the weather service right now."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise SkillExecutionError(
                f"Weather service returned HTTP {exc.response.status_code}."
            ) from exc

        return (
            f"location={display_name}; "
            f"condition={current['condition']}; "
            f"temperature={current['temperature']}{current['temperature_unit']}; "
            f"feels_like={current['feels_like']}{current['temperature_unit']}; "
            f"humidity={current['humidity']}%; "
            f"wind={current['wind_speed']} {current['wind_unit']}"
        )

    async def _geocode(self, location: str) -> tuple[float, float, str]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(
                _GEOCODE_URL,
                params={"name": location, "count": 1, "language": "en", "format": "json"},
            )
            response.raise_for_status()
            payload = response.json()

        results = payload.get("results") or []
        if not results:
            raise ValueError(f"Location not found: {location}")

        match = results[0]
        name = match.get("name", location)
        country = match.get("country", "")
        display_name = f"{name}, {country}".strip(", ")
        return float(match["latitude"]), float(match["longitude"]), display_name

    async def _fetch_current(self, lat: float, lon: float) -> dict[str, Any]:
        temp_unit = "fahrenheit" if self._units == "fahrenheit" else "celsius"
        wind_unit = "mph" if self._units == "fahrenheit" else "kmh"

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(
                _FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": (
                        "temperature_2m,apparent_temperature,"
                        "relative_humidity_2m,wind_speed_10m,weather_code"
                    ),
                    "temperature_unit": temp_unit,
                    "wind_speed_unit": wind_unit,
                },
            )
            response.raise_for_status()
            payload = response.json()

        current = payload.get("current") or {}
        return {
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "condition": _wmo_condition(int(current.get("weather_code", 0))),
            "temperature_unit": "°F" if temp_unit == "fahrenheit" else "°C",
            "wind_unit": "mph" if wind_unit == "mph" else "km/h",
        }
