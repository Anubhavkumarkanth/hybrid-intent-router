"""Task actions with a fake HTTP layer (no network)."""

import datetime as dt

import pytest

from router import actions as A

GEO = {
    "Bengaluru": {
        "name": "Bengaluru",
        "country": "India",
        "latitude": 12.97,
        "longitude": 77.59,
        "timezone": "Asia/Kolkata",
    },
    "London": {
        "name": "London",
        "country": "United Kingdom",
        "latitude": 51.5,
        "longitude": -0.12,
        "timezone": "Europe/London",
    },
    "Mumbai": {"name": "Mumbai", "country": "India", "latitude": 19.07, "longitude": 72.88, "timezone": "Asia/Kolkata"},
}


@pytest.fixture(autouse=True)
def fake_http(monkeypatch):
    calls = []

    def fetch(url):
        calls.append(url)
        if "geocoding" in url:
            name = url.split("name=")[1].split("&")[0].replace("+", " ")
            hit = GEO.get(name)
            return {"results": [hit]} if hit else {}
        if "forecast" in url:
            return {
                "current": {
                    "temperature_2m": 24.4,
                    "relative_humidity_2m": 60,
                    "weather_code": 2,
                    "wind_speed_10m": 9.0,
                },
                "daily": {
                    "temperature_2m_min": [19.0],
                    "temperature_2m_max": [28.0],
                    "precipitation_probability_max": [20],
                },
            }
        if "frankfurter" in url:
            return {"date": "2026-11-20", "rates": {url.split("symbols=")[1]: 92.5}}
        if "dictionaryapi" in url:
            return [{"meanings": [{"partOfSpeech": "noun", "definitions": [{"definition": "a happy accident"}]}]}]
        raise AssertionError(url)

    for cache in (A._geo_cache, A._weather_cache, A._fx_cache, A._dict_cache):
        cache.data.clear()
    monkeypatch.setattr(A, "fetch", fetch)
    return calls


def test_weather_uses_named_city_or_default():
    r = A.run_action("weather_query", "what's the weather in London today")
    assert r["kind"] == "weather" and r["title"].startswith("London") and r["detail"]["place_from_message"]
    assert "24°C" in r["reply"] and "partly cloudy" in r["reply"] and "20% chance of rain" in r["reply"]
    r = A.run_action("weather_query", "will it rain")
    assert r["title"].startswith("Bengaluru") and r["detail"]["place_from_message"] is False


def test_weather_understands_hindi_and_kannada_city_names():
    assert A.run_action("weather_query", "मुंबई में मौसम कैसा है")["title"].startswith("Mumbai")
    assert A.run_action("weather_query", "ಬೆಂಗಳೂರಿನ ಹವಾಮಾನ")["title"].startswith("Bengaluru")


def test_time_in_ist_and_named_city():
    now = dt.datetime(2026, 11, 22, 10, 0, tzinfo=dt.timezone.utc)
    assert A.current_time("what time is it", now)["reply"] == "It's 3:30 PM on Sunday, 22 November 2026 in India (IST)."
    assert A.current_time("what time is it in London", now)["detail"]["time"] == "10:00 AM"


@pytest.mark.parametrize(
    "text,src,dst,amount",
    [
        ("one dollar equals how much inr", "USD", "INR", 1.0),
        ("convert 250 euros to rupees", "EUR", "INR", 250.0),
        ("how many dollars is 1000 rupees", "USD", "INR", 1000.0),
        ("100 डॉलर कितने रुपये", "USD", "INR", 100.0),
        ("what is the exchange rate of pound", "GBP", "INR", 1.0),
    ],
)
def test_currency_parsing(text, src, dst, amount):
    d = A.run_action("qa_currency", text)["detail"]
    assert (d["from"], d["to"], d["amount"]) == (src, dst, amount)
    assert d["value"] == round(amount * 92.5, 2)


@pytest.mark.parametrize(
    "text,value",
    [
        ("what is 25 times 4", 100),
        ("12 plus 30 divided by 3", 22),
        ("calculate (2 + 3) * 4", 20),
        ("what is 7 squared", 49),
    ],
)
def test_maths(text, value):
    assert A.run_action("qa_maths", text)["detail"]["value"] == value


def test_maths_refuses_non_arithmetic_and_huge_powers():
    assert A.run_action("qa_maths", "what is the meaning of life") is None
    assert A.run_action("qa_maths", "2 ** 999999") is None
    assert A.run_action("qa_maths", "__import__('os')") is None


def test_definition_and_joke():
    assert A.run_action("qa_definition", "define serendipity")["reply"] == "serendipity (noun): a happy accident"
    assert A.run_action("general_joke", "tell me a joke")["reply"] in A.JOKES


def test_no_action_for_other_intents_and_api_failure_is_safe(monkeypatch):
    assert A.run_action("alarm_set", "wake me up at 5") is None
    assert A.describe("alarm_set") == "set an alarm"

    def broken(url):
        raise OSError("network down")

    monkeypatch.setattr(A, "fetch", broken)
    r = A.run_action("weather_query", "weather in Paris")
    assert r["detail"].get("error") and "right now" in r["reply"]
