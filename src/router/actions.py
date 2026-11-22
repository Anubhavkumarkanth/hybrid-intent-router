"""Do the task for some intents, with free public APIs that need no key.

    weather_query   current weather (Open-Meteo)            datetime_query  current time, IST or a named city
    qa_currency     currency conversion (Frankfurter, ECB)  qa_maths        arithmetic, safely evaluated
    qa_definition   English word meaning (dictionaryapi.dev) general_joke   a short built-in joke

run_action(intent, text) returns {"kind", "title", "reply", "detail", "source"} or None. Every other intent gets
a plain confirmation from describe(). Network calls have short timeouts and small caches; any failure returns
an answer that says so, never an exception, so a slow API can't break the main prediction.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import operator
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

TIMEOUT_S = 4.0
DEFAULT_CITY = "Bengaluru"


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "hybrid-intent-router-demo"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


fetch = _http_json  # tests replace this with a fake


class TTLCache:
    def __init__(self, seconds: float):
        self.seconds, self.data, self.lock = seconds, {}, threading.Lock()

    def get(self, key, make):
        now = time.monotonic()
        with self.lock:
            hit = self.data.get(key)
            if hit and now - hit[0] < self.seconds:
                return hit[1]
        value = make()
        with self.lock:
            self.data[key] = (now, value)
        return value


_geo_cache, _weather_cache, _fx_cache, _dict_cache = TTLCache(86400), TTLCache(600), TTLCache(3600), TTLCache(86400)

# ------------------------------------------------------------------ places

# Open-Meteo's place search only understands Latin-script names, so common Indian cities written in Devanagari
# or Kannada script are mapped first.
CITY_ALIASES = {
    "बेंगलुरु": "Bengaluru",
    "बैंगलोर": "Bengaluru",
    "मुंबई": "Mumbai",
    "दिल्ली": "Delhi",
    "चेन्नई": "Chennai",
    "कोलकाता": "Kolkata",
    "हैदराबाद": "Hyderabad",
    "पुणे": "Pune",
    "जयपुर": "Jaipur",
    "लखनऊ": "Lucknow",
    "अहमदाबाद": "Ahmedabad",
    "पटना": "Patna",
    "भोपाल": "Bhopal",
    "मैसूर": "Mysuru",
    "लंदन": "London",
    "ಬೆಂಗಳೂರು": "Bengaluru",
    "ಬೆಂಗಳೂರಿನ": "Bengaluru",
    "ಮೈಸೂರು": "Mysuru",
    "ಮೈಸೂರಿನ": "Mysuru",
    "ಮಂಗಳೂರು": "Mangaluru",
    "ಹುಬ್ಬಳ್ಳಿ": "Hubballi",
    "ಮುಂಬೈ": "Mumbai",
    "ದೆಹಲಿ": "Delhi",
    "ಚೆನ್ನೈ": "Chennai",
    "ಹೈದರಾಬಾದ್": "Hyderabad",
}
_STOP = {
    "today",
    "tomorrow",
    "tonight",
    "now",
    "right",
    "this",
    "next",
    "the",
    "week",
    "weekend",
    "morning",
    "evening",
    "please",
    "currently",
    "be",
    "is",
    "like",
    "going",
}


def _candidates(text: str) -> list[str]:
    out = [en for native, en in CITY_ALIASES.items() if native in text]
    for m in re.finditer(r"\b(?:in|at|for|of)\s+([A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){0,2})", text, re.I):
        words = []
        for w in m.group(1).split():
            if w.lower() in _STOP:
                break
            words.append(w)
        if words:
            out.append(" ".join(words))
    out += [w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text) if w.lower() not in _STOP]
    seen, uniq = set(), []
    for c in out:
        if c.lower() not in seen:
            seen.add(c.lower())
            uniq.append(c)
    return uniq


def geocode(name: str) -> dict | None:
    def make():
        q = urllib.parse.urlencode({"name": name, "count": 1, "language": "en"})
        res = (fetch(f"https://geocoding-api.open-meteo.com/v1/search?{q}") or {}).get("results") or []
        if not res:
            return None
        r = res[0]
        return {
            "name": r["name"],
            "country": r.get("country", ""),
            "lat": r["latitude"],
            "lon": r["longitude"],
            "tz": r.get("timezone", "UTC"),
        }

    return _geo_cache.get(name.lower(), make)


def find_place(text: str) -> tuple[dict | None, bool]:
    """(place, found_in_text). Falls back to the default city when the message names none."""
    for cand in _candidates(text):
        place = geocode(cand)
        if place:
            return place, True
    return geocode(DEFAULT_CITY), False


# ------------------------------------------------------------------ weather

WMO = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm with hail",
    99: "a thunderstorm with hail",
}


def weather(text: str) -> dict:
    place, named = find_place(text)
    if not place:
        return _fail("weather", "I couldn't look up the weather right now.")

    def make():
        q = urllib.parse.urlencode(
            {
                "latitude": place["lat"],
                "longitude": place["lon"],
                "timezone": "auto",
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": 1,
            }
        )
        return fetch(f"https://api.open-meteo.com/v1/forecast?{q}")

    w = _weather_cache.get((round(place["lat"], 2), round(place["lon"], 2)), make)
    cur, day = w["current"], w["daily"]
    desc = WMO.get(cur["weather_code"], "mixed weather")
    rain = (day.get("precipitation_probability_max") or [None])[0]
    where = f"{place['name']}, {place['country']}".strip(", ")
    reply = (
        f"It's {cur['temperature_2m']:.0f}°C with {desc} in {place['name']} right now. Today: "
        f"{day['temperature_2m_min'][0]:.0f} to {day['temperature_2m_max'][0]:.0f}°C"
        + (f", {rain}% chance of rain." if rain is not None else ".")
    )
    return {
        "kind": "weather",
        "title": where,
        "reply": reply,
        "detail": {
            "temp_c": cur["temperature_2m"],
            "description": desc,
            "humidity": cur["relative_humidity_2m"],
            "wind_kmh": cur["wind_speed_10m"],
            "min_c": day["temperature_2m_min"][0],
            "max_c": day["temperature_2m_max"][0],
            "rain_pct": rain,
            "code": cur["weather_code"],
            "place_from_message": named,
        },
        "source": "Open-Meteo",
    }


# ------------------------------------------------------------------ time


def current_time(text: str, now: dt.datetime | None = None) -> dict:
    place, named = find_place(text) if _candidates(text) else (None, False)
    tz_name, where = (place["tz"], place["name"]) if (place and named) else ("Asia/Kolkata", "India (IST)")
    t = (now or dt.datetime.now(dt.timezone.utc)).astimezone(ZoneInfo(tz_name))
    clock = t.strftime("%I:%M %p").lstrip("0")
    day = f"{t.strftime('%A')}, {t.day} {t.strftime('%B %Y')}"
    return {
        "kind": "time",
        "title": where,
        "reply": f"It's {clock} on {day} in {where}.",
        "detail": {"time": clock, "date": day, "timezone": tz_name},
        "source": "server clock",
    }


# ------------------------------------------------------------------ currency

CURRENCY_WORDS = {
    "USD": ["usd", "dollar", "dollars", "dolla", "$", "डॉलर", "ಡಾಲರ್"],
    "INR": ["inr", "rupee", "rupees", "rs", "₹", "रुपये", "रुपया", "रुपए", "ರೂಪಾಯಿ", "ರೂಪಾಯಿಗಳು"],
    "EUR": ["eur", "euro", "euros", "€", "यूरो", "ಯೂರೋ", "ಯುರೋ"],
    "GBP": ["gbp", "pound", "pounds", "£", "पाउंड", "ಪೌಂಡ್"],
    "JPY": ["jpy", "yen", "¥", "येन", "ಯೆನ್"],
    "AUD": ["aud"],
    "CAD": ["cad"],
    "SGD": ["sgd"],
    "CNY": ["cny", "yuan"],
    "CHF": ["chf", "franc", "francs"],
}
NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "ten": 10,
    "twenty": 20,
    "fifty": 50,
    "hundred": 100,
    "thousand": 1000,
    "एक": 1,
    "दो": 2,
    "सौ": 100,
    "ಒಂದು": 1,
    "ಹತ್ತು": 10,
}


def _find_currencies(text: str) -> list[str]:
    low = text.lower()
    hits = []
    for code, words in CURRENCY_WORDS.items():
        for w in words:
            pattern = re.escape(w) if not w.isalpha() or not w.isascii() else rf"\b{re.escape(w)}\b"
            m = re.search(pattern, low)
            if m:
                hits.append((m.start(), code))
                break
    return [code for _, code in sorted(hits)]


def _find_amount(text: str) -> float:
    m = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    for w in re.findall(r"\w+", text.lower()):
        if w in NUMBER_WORDS:
            return float(NUMBER_WORDS[w])
    return 1.0


def currency(text: str) -> dict:
    codes = _find_currencies(text)
    src, dst = (codes + ["USD", "INR"])[:2] if codes else ("USD", "INR")
    if len(codes) == 1:
        src, dst = codes[0], ("INR" if codes[0] != "INR" else "USD")
    if src == dst:
        dst = "INR" if src != "INR" else "USD"
    amount = _find_amount(text)
    try:
        data = _fx_cache.get(
            (src, dst), lambda: fetch(f"https://api.frankfurter.dev/v1/latest?base={src}&symbols={dst}")
        )
        rate = data["rates"][dst]
    except Exception:
        return _fail("currency", "I couldn't get exchange rates right now.")
    value = amount * rate
    return {
        "kind": "currency",
        "title": f"{src} → {dst}",
        "reply": f"{amount:,.2f} {src} is about {value:,.2f} {dst} (rate {rate:,.4f}, {data.get('date', '')}).",
        "detail": {
            "amount": amount,
            "from": src,
            "to": dst,
            "rate": rate,
            "value": round(value, 2),
            "date": data.get("date"),
        },
        "source": "Frankfurter (European Central Bank rates)",
    }


# ------------------------------------------------------------------ maths

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_WORD_OPS = [
    (r"\bmultiplied by\b|\btimes\b|\binto\b|\bx\b|×", "*"),
    (r"\bdivided by\b|\bover\b|÷", "/"),
    (r"\bplus\b|\badd\b|\band\b", "+"),
    (r"\bminus\b|\bless\b", "-"),
    (r"\bsquared\b", "**2"),
    (r"\bto the power of\b|\bpower\b", "**"),
    (r"\bpercent of\b|% of", "/100*"),
]


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("not arithmetic")


def maths(text: str) -> dict | None:
    expr = text.lower()
    for pat, rep in _WORD_OPS:
        expr = re.sub(pat, rep, expr)
    m = re.search(r"[-(\d][\d\s.+\-*/%()]*\d\)?", expr)
    if not m or not re.search(r"[+\-*/%]", m.group(0).strip().lstrip("-")) or len(m.group(0)) > 80:
        return None
    shown = re.sub(r"\s+", " ", m.group(0)).strip()
    try:
        value = _safe_eval(ast.parse(shown, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    value = round(value, 6) if isinstance(value, float) else value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    pretty = shown.replace("**", "^").replace("*", "×").replace("/", "÷")
    return {
        "kind": "maths",
        "title": pretty,
        "reply": f"{pretty} = {value:,}" if isinstance(value, int) else f"{pretty} = {value}",
        "detail": {"expression": shown, "value": value},
        "source": "calculator",
    }


# ------------------------------------------------------------------ definition


def definition(text: str) -> dict | None:
    m = re.search(
        r"(?:define|definition of|meaning of|meaning|what does)\s+(?:the word\s+)?['\"]?([A-Za-z-]{2,30})", text, re.I
    ) or re.search(r"\b([A-Za-z-]{3,30})\s+(?:means?|meaning)\b", text, re.I)
    if not m:
        return None
    word = m.group(1).lower()
    try:
        data = _dict_cache.get(
            word, lambda: fetch(f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}")
        )
        meaning = data[0]["meanings"][0]
        text_def = meaning["definitions"][0]["definition"]
    except Exception:
        return _fail("definition", f'I couldn\'t find a definition for "{word}".')
    return {
        "kind": "definition",
        "title": word,
        "reply": f"{word} ({meaning.get('partOfSpeech', '')}): {text_def}",
        "detail": {"word": word, "part_of_speech": meaning.get("partOfSpeech"), "definition": text_def},
        "source": "Free Dictionary API",
    }


# ------------------------------------------------------------------ joke

JOKES = [
    "Why did the developer go broke? Because he used up all his cache.",
    "I told my computer I needed a break, and it said: no problem, I'll go to sleep.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "Why was the maths book sad? It had too many problems.",
    "What do you call a fake noodle? An impasta.",
    "Why don't scientists trust atoms? Because they make up everything.",
    "I would tell you a UDP joke, but you might not get it.",
]


def joke(text: str) -> dict:
    j = random.choice(JOKES)
    return {"kind": "joke", "title": "A joke", "reply": j, "detail": {}, "source": "built-in list"}


# ------------------------------------------------------------------ dispatch


def _fail(kind: str, reply: str) -> dict:
    return {"kind": kind, "title": "Not available", "reply": reply, "detail": {"error": True}, "source": ""}


HANDLERS = {
    "weather_query": weather,
    "datetime_query": current_time,
    "qa_currency": currency,
    "qa_maths": maths,
    "qa_definition": definition,
    "general_joke": joke,
}

DESCRIPTIONS = {
    "alarm_set": "set an alarm",
    "alarm_query": "check your alarms",
    "alarm_remove": "remove an alarm",
    "audio_volume_mute": "mute the sound",
    "audio_volume_up": "turn the volume up",
    "audio_volume_down": "turn the volume down",
    "audio_volume_other": "change the volume",
    "calendar_set": "add something to your calendar",
    "calendar_query": "check your calendar",
    "calendar_remove": "remove a calendar event",
    "cooking_recipe": "find a recipe",
    "cooking_query": "answer a cooking question",
    "email_sendemail": "send an email",
    "email_query": "check your email",
    "email_querycontact": "look up a contact",
    "email_addcontact": "add a contact",
    "iot_hue_lightoff": "turn the lights off",
    "iot_hue_lighton": "turn the lights on",
    "iot_hue_lightchange": "change the light colour",
    "iot_hue_lightdim": "dim the lights",
    "iot_hue_lightup": "brighten the lights",
    "iot_cleaning": "start the vacuum cleaner",
    "iot_coffee": "make coffee",
    "iot_wemo_on": "switch the smart plug on",
    "iot_wemo_off": "switch the smart plug off",
    "lists_query": "read a list",
    "lists_createoradd": "add to a list",
    "lists_remove": "remove from a list",
    "music_query": "tell you about the music",
    "music_likeness": "remember that you like this song",
    "music_dislikeness": "remember that you don't like this song",
    "music_settings": "change music settings",
    "news_query": "get the news",
    "play_music": "play music",
    "play_radio": "play the radio",
    "play_podcasts": "play a podcast",
    "play_audiobook": "play an audiobook",
    "play_game": "start a game",
    "recommendation_events": "suggest events",
    "recommendation_locations": "suggest places",
    "recommendation_movies": "suggest movies",
    "social_post": "post on social media",
    "social_query": "check social media",
    "takeaway_order": "order food",
    "takeaway_query": "check a food order",
    "transport_ticket": "book a ticket",
    "transport_taxi": "book a taxi",
    "transport_query": "check transport",
    "transport_traffic": "check the traffic",
    "datetime_convert": "convert a time between zones",
    "general_greet": "say hello",
    "general_quirky": "chat",
    "qa_factoid": "answer a general question",
    "qa_stock": "check a stock price",
    "weather_query": "check the weather",
    "datetime_query": "tell the time",
    "qa_currency": "convert currency",
    "qa_maths": "do the maths",
    "qa_definition": "define a word",
    "general_joke": "tell a joke",
}


def describe(intent: str) -> str:
    return DESCRIPTIONS.get(intent, intent.replace("_", " "))


def run_action(intent: str, text: str) -> dict | None:
    handler = HANDLERS.get(intent)
    if not handler:
        return None
    try:
        return handler(text)
    except Exception:  # an outside API changed or is down: say so, never break the prediction
        return _fail(intent, "That service is not answering right now.")
