"""tool-distractor-selection-v1 â€” precise tool selection among distractors.

World: a 12-tool pool (weather, FX, email, calendar, translation, music,
flights, taxis, social, smart-home, recipes, stocks) where only the 2 relevant
tools may be called. Inspired by BFCL irrelevance detection. Teaches: reading
tool descriptions, refusing irrelevant tools, minimal call count.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains

WEATHER = {
    "tokyo": {"temp_c": 18, "condition": "clear"},
    "oslo": {"temp_c": -3, "condition": "snow"},
    "lima": {"temp_c": 22, "condition": "humid"},
    "cairo": {"temp_c": 34, "condition": "hot"},
}
RATES = {"EUR_USD": 1.09, "GBP_JPY": 191.4, "USD_CHF": 0.88, "AUD_CAD": 0.91}


def _make_tools():
    def get_weather(state, args):
        city = str(args.get("city", "")).lower()
        if city not in WEATHER:
            return None, f"ERROR: unknown city {city}"
        state["weather"] = city
        data = WEATHER[city]
        return data, f"{city}: {data['temp_c']}C, {data['condition']}"

    def get_rate(state, args):
        pair = str(args.get("pair", "")).upper()
        if pair not in RATES:
            return None, f"ERROR: unknown pair {pair}"
        state["rate"] = RATES[pair]
        return RATES[pair], f"{pair} = {RATES[pair]}"

    def irrelevant(name, result):
        def fn(state, args):
            state[f"irrelevant_{name}"] = True
            return result, f"{name} executed (irrelevant to this task)"
        return fn

    return {
        "get_weather": ToolSpec("get_weather", "Current weather for a city.", {"city": "str"}, get_weather),
        "get_exchange_rate": ToolSpec("get_exchange_rate", "FX rate for a currency pair.", {"pair": "str"}, get_rate),
        "send_email": ToolSpec("send_email", "Send an email.", {"to": "str", "body": "str"}, lambda s, a: (None, "sent (irrelevant)"), distractor=True),
        "create_event": ToolSpec("create_event", "Create a calendar event.", {"title": "str"}, lambda s, a: (None, "created (irrelevant)"), distractor=True),
        "translate_text": ToolSpec("translate_text", "Translate text.", {"text": "str"}, lambda s, a: ("ciao", "translated (irrelevant)"), distractor=True),
        "play_music": ToolSpec("play_music", "Play a track.", {"track": "str"}, lambda s, a: (None, "playing (irrelevant)"), distractor=True),
        "book_flight": ToolSpec("book_flight", "Book a flight.", {"origin": "str", "to": "str"}, lambda s, a: (None, "booked (irrelevant)"), distractor=True),
        "order_taxi": ToolSpec("order_taxi", "Hail a taxi.", {"city": "str"}, lambda s, a: (None, "taxi ordered (irrelevant)"), distractor=True),
        "post_social": ToolSpec("post_social", "Post to social media.", {"text": "str"}, lambda s, a: (None, "posted (irrelevant)"), distractor=True),
        "toggle_lights": ToolSpec("toggle_lights", "Toggle smart lights.", {"room": "str"}, lambda s, a: (None, "toggled (irrelevant)"), distractor=True),
        "search_recipes": ToolSpec("search_recipes", "Find recipes.", {"query": "str"}, lambda s, a: ([], "no recipes (irrelevant)"), distractor=True),
        "get_stock_quote": ToolSpec("get_stock_quote", "Stock price lookup.", {"ticker": "str"}, lambda s, a: (None, "quote (irrelevant)"), distractor=True),
    }


def _select_task(idx, name, city, pair):
    tools = _make_tools()

    def goal(state):
        return state.get("weather") == city and state.get("rate") == RATES[pair], (
            f"weather={state.get('weather')}, rate={state.get('rate')}"
        )

    spec = (
        "You have 12 tools available: get_weather, get_exchange_rate, send_email, "
        "create_event, translate_text, play_music, book_flight, order_taxi, "
        "post_social, toggle_lights, search_recipes, get_stock_quote. Only some are "
        "relevant to a given task â€” calling irrelevant tools costs score.\n\n"
        f"Task: report the current temperature in {city} AND the {pair} exchange "
        "rate. Use ONLY the relevant tools, as few calls as possible, and include "
        "both values in your <answer>."
    )
    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init={},
        goal=goal,
        expert=[("get_weather", {"city": city}), ("get_exchange_rate", {"pair": pair})],
        verify_answer=answer_contains(str(WEATHER[city]["temp_c"]), str(RATES[pair])),
        expert_answer=f"{city} {WEATHER[city]['temp_c']}C, {pair} = {RATES[pair]}",
        expected_concepts=["get_weather", "get_exchange_rate", city, pair],
        scenario="distractor-selection",
    )


TASKS = [
    _select_task(0, "weather_and_fx_tokyo", "tokyo", "EUR_USD"),
    _select_task(1, "weather_and_fx_oslo", "oslo", "GBP_JPY"),
    _select_task(2, "weather_and_fx_lima", "lima", "USD_CHF"),
    _select_task(3, "weather_and_fx_cairo", "cairo", "AUD_CAD"),
]
