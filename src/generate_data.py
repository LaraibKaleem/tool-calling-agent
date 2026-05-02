#!/usr/bin/env python3
import json, random, hashlib, argparse, pathlib, os
from datetime import date, timedelta
from typing import Optional

random.seed(42)

SYSTEM_PROMPT = """You are a mobile assistant. When the user's request maps to one of the available tools, respond ONLY with a tool call in this exact format (no other text):
<tool_call>{"tool": "TOOL_NAME", "args": {ARGS_JSON}}</tool_call>

Available tools:
- weather:  {"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}
- calendar: {"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}
- convert:  {"tool": "convert",  "args": {"value": number, "from_unit": "string", "to_unit": "string"}}
- currency: {"tool": "currency", "args": {"amount": number, "from": "ISO3", "to": "ISO3"}}
- sql:      {"tool": "sql",      "args": {"query": "string"}}

If no tool fits, respond naturally in plain text."""

CITIES = [
    "London","Paris","Tokyo","New York","Sydney","Dubai","Berlin","Toronto",
    "Singapore","Mumbai","Cairo","Moscow","Beijing","Seoul","Istanbul",
    "Amsterdam","Rome","Madrid","Bangkok","Lagos","Nairobi","Buenos Aires",
    "São Paulo","Mexico City","Jakarta","Karachi","Lahore","Rawalpindi",
    "Dhaka","Colombo","Kathmandu","Tehran","Baghdad","Riyadh","Casablanca",
    "Helsinki","Oslo","Copenhagen","Stockholm","Vienna","Zurich","Prague",
    "Warsaw","Athens","Lisbon","Brussels","Dublin","Los Angeles","Chicago",
    "Houston","Miami","Seattle","Denver","Vancouver","Montreal","Melbourne",
]

CURRENCY_PAIRS = [
    ("USD","EUR"),("EUR","USD"),("USD","GBP"),("GBP","USD"),("USD","JPY"),
    ("JPY","USD"),("EUR","GBP"),("GBP","EUR"),("USD","CAD"),("CAD","USD"),
    ("USD","AUD"),("AUD","USD"),("EUR","CHF"),("CHF","EUR"),("USD","INR"),
    ("INR","USD"),("USD","CNY"),("CNY","USD"),("USD","MXN"),("MXN","USD"),
    ("USD","BRL"),("BRL","USD"),("EUR","JPY"),("GBP","JPY"),("USD","KRW"),
    ("USD","SGD"),("USD","HKD"),("USD","NOK"),("USD","SEK"),("USD","PKR"),
    ("USD","EGP"),("USD","SAR"),("USD","AED"),("EUR","AED"),("USD","TRY"),
    ("USD","ZAR"),("GBP","AUD"),("EUR","AUD"),("USD","NZD"),("USD","THB"),
]

WEIGHT_UNITS = [
    ("kilograms","pounds"),("pounds","kilograms"),("grams","ounces"),
    ("ounces","grams"),("stones","kilograms"),("kilograms","stones"),
]

LENGTH_UNITS = [
    ("miles","kilometers"),("kilometers","miles"),("meters","feet"),
    ("feet","meters"),("inches","centimeters"),("centimeters","inches"),
    ("yards","meters"),("meters","yards"),
]

TEMP_UNITS = [
    ("celsius","fahrenheit"),("fahrenheit","celsius"),("celsius","kelvin"),
    ("kelvin","celsius"),
]

VOLUME_UNITS = [
    ("gallons","liters"),("liters","gallons"),("cups","milliliters"),
    ("milliliters","cups"),("pints","liters"),("liters","pints"),
]

ALL_UNIT_PAIRS = WEIGHT_UNITS + LENGTH_UNITS + TEMP_UNITS + VOLUME_UNITS

SQL_TEMPLATES = [
    "SELECT * FROM {table} WHERE {col} > {val}",
    "SELECT * FROM {table} WHERE {col} = '{sval}'",
    "SELECT COUNT(*) FROM {table}",
    "SELECT AVG({col}) FROM {table}",
    "SELECT {col}, COUNT(*) FROM {table} GROUP BY {col}",
    "SELECT * FROM {table} ORDER BY {col} DESC LIMIT {val}",
    "SELECT DISTINCT {col} FROM {table}",
    "SELECT * FROM {table} WHERE {col} IS NULL",
]

TABLES = ["users","orders","products","customers","employees","inventory","transactions","logs"]
COLS   = ["age","price","salary","quantity","status","name","email","category","country","score"]
SVALS  = ["active","pending","USA","UK","admin","completed","cancelled","premium"]

WEATHER_TEMPLATES = [
    "What's the weather in {city}?",
    "Weather in {city} please",
    "How hot is it in {city}?",
    "How cold is it in {city}?",
    "Current weather in {city}?",
    "Tell me the weather in {city}",
    "What's the temperature in {city}?",
    "Is it warm in {city} today?",
    "Weather update for {city}",
    "Check weather for {city}",
    "What's the weather in {city}? Give me {unit_word}",
    "Weather in {city} in {unit_letter}",
    "{city} weather in {unit_word}",
]

WEATHER_UNIT_WORDS = {
    "C": ["Celsius","celsius","centigrade","metric"],
    "F": ["Fahrenheit","fahrenheit","imperial","the American unit"],
}

CAL_LIST_TEMPLATES = [
    "Show my calendar for {date}",
    "What's on my schedule for {date}?",
    "What events do I have on {date}?",
    "Show me my appointments on {date}",
    "What meetings are on {date}?",
    "Check my calendar for {date}",
    "List my events for {date}",
    "Any appointments on {date}?",
]

CAL_CREATE_TEMPLATES = [
    "Add {title} to my calendar on {date}",
    "Schedule {title} on {date}",
    "Book {title} for {date}",
    "Create a {title} event on {date}",
    "Put {title} on my calendar for {date}",
    "Set up {title} on {date}",
    "Add an event called {title} for {date}",
]

EVENT_TITLES = [
    "dentist appointment","doctor visit","team meeting","birthday party",
    "lunch with Sarah","job interview","gym session","project deadline",
    "quarterly review","client call","standup meeting","conference",
    "workshop","yoga class","coding bootcamp","product launch",
    "board meeting","performance review","sprint planning","hackathon",
]

CONVERT_TEMPLATES = [
    "Convert {value} {from_unit} to {to_unit}",
    "How many {to_unit} is {value} {from_unit}?",
    "{value} {from_unit} to {to_unit}",
    "What is {value} {from_unit} in {to_unit}?",
    "Turn {value} {from_unit} into {to_unit}",
    "Change {value} {from_unit} to {to_unit}",
    "I have {value} {from_unit}, what is that in {to_unit}?",
    "{value} {from_unit} equals how many {to_unit}?",
]

CURRENCY_TEMPLATES = [
    "Convert {amount} {from_} to {to}",
    "How much is {amount} {from_name} in {to_name}?",
    "Exchange {amount} {from_} to {to}",
    "{amount} {from_} in {to}?",
    "What is {amount} {from_} in {to}?",
    "Change {amount} {from_name} into {to_name}",
    "Turn {amount} {from_} into {to}",
    "{amount} {from_} equals how many {to}?",
]

CURRENCY_NAMES = {
    "USD":"US dollars","EUR":"euros","GBP":"British pounds","JPY":"Japanese yen",
    "CAD":"Canadian dollars","AUD":"Australian dollars","CHF":"Swiss francs",
    "INR":"Indian rupees","CNY":"Chinese yuan","MXN":"Mexican pesos",
    "BRL":"Brazilian reais","KRW":"South Korean won","SGD":"Singapore dollars",
    "PKR":"Pakistani rupees","EGP":"Egyptian pounds","SAR":"Saudi riyals",
    "AED":"UAE dirhams","TRY":"Turkish lira","ZAR":"South African rand",
    "NZD":"New Zealand dollars","THB":"Thai baht",
}

SQL_PROMPT_TEMPLATES = [
    "Run this SQL: {query}",
    "Execute: {query}",
    "Query the database: {query}",
    "Run query: {query}",
    "Execute this SQL query: {query}",
    "Database query: {query}",
    "Please run: {query}",
    "SQL: {query}",
]

CHITCHAT_PROMPTS = [
    "How are you doing?","What's your name?","Tell me a joke",
    "Who invented the internet?","What's the meaning of life?",
    "Can you write me a poem?","What's 2 + 2?","Recommend a good movie",
    "What do you think about AI?","Help me write an email",
    "What's the capital of Japan?","Tell me about space exploration",
]

MISSING_TOOL_PROMPTS = [
    "Send a text message to mom","Play some music","Take a photo",
    "Turn on the lights","Order a pizza","Call a taxi",
    "Send an email to John","Post this to Twitter","Book a flight to Paris",
]

AMBIGUOUS_PROMPTS = [
    "Convert that to euros","What's the weather there?",
    "Change it to Fahrenheit","Add it to my calendar",
    "Convert that amount","Make it dollars",
]

REFUSAL_RESPONSES = [
    "I can help with weather, calendar, unit conversion, currency exchange, and SQL queries. That request doesn't match any of those tools.",
    "That's outside what I can help with. My available tools are: weather, calendar, unit conversion, currency exchange, and SQL.",
    "I don't have a tool for that. I can assist with weather, calendar events, conversions, currency exchange, and database queries.",
]

AMBIGUOUS_RESPONSES = [
    "I'd need more context for that. Could you clarify what you'd like to convert/check/schedule?",
    "I'm not sure what you're referring to. Could you provide more details?",
    "I don't have enough context. Could you be more specific?",
]

CODE_SWITCH_PREFIXES = [
    "Bhai ","Yaar ","Mujhe batao ","Ji ","Por favor ","Oye ",
    "Please bhai ","Yaar bata ","Ap mujhe batao ",
]

CODE_SWITCH_PREFIXES = [
    "Bhai ","Yaar ","Mujhe batao ","Ji ","Por favor ","Oye ",
    "Please bhai ","Yaar bata ","Ap mujhe batao ",
]

HINDI_URDU_WEATHER = [
    ("Mujhe {city} ka mausam batao Celsius mein",    "C"),
    ("Mujhe {city} ka mausam batao Fahrenheit mein", "F"),
    ("Ap mujhe {city} ka mausam batao",              "C"),
    ("Bhai {city} mein kitni garmi hai",             "C"),
    ("Bhai {city} mein kitni sardi hai",             "C"),
    ("Yaar {city} ka mausam kaisa hai",              "C"),
    ("{city} mein aaj mausam kaisa hai",             "C"),
    ("Mujhe {city} ka temperature batao Celsius",    "C"),
    ("Mujhe {city} ka temperature batao Fahrenheit", "F"),
    ("Bhai {city} mein kitni garmi hai Fahrenheit",  "F"),
    ("Ap batao {city} mein kaisa mausam hai",        "C"),
    ("Ji {city} ka mausam kya hai aaj",              "C"),
    ("{city} ka mausam batao",                       "C"),
    ("Aaj {city} mein mausam kaisa hai",             "C"),
    ("Bhai zara {city} ka mausam check karo",        "C"),
]

def random_date() -> str:
    base = date(2024, 1, 1)
    return (base + timedelta(days=random.randint(0, 730))).strftime("%Y-%m-%d")

def random_amount(low=1, high=10000) -> float:
    v = random.uniform(low, high)
    return round(v, 2) if random.random() < 0.3 else int(v)

def random_sql() -> str:
    t  = random.choice(SQL_TEMPLATES)
    t1 = random.choice(TABLES)
    c1 = random.choice(COLS)
    v1 = random.randint(1, 1000)
    sv = random.choice(SVALS)
    return t.format(table=t1, col=c1, val=v1, sval=sv)

def mk_tool_call(tool: str, args: dict) -> str:
    return f'<tool_call>{json.dumps({"tool": tool, "args": args}, ensure_ascii=False)}</tool_call>'

def add_typos(text: str, rate: float = 0.15) -> str:
    words = text.split()
    out = []
    for w in words:
        if random.random() < rate and len(w) > 3:
            i = random.randint(1, len(w)-2)
            op = random.choice(["swap","drop","double"])
            if op == "swap":
                wl = list(w); wl[i], wl[i-1] = wl[i-1], wl[i]; w = "".join(wl)
            elif op == "drop":
                w = w[:i] + w[i+1:]
            elif op == "double":
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)

def gen_weather():
    city = random.choice(CITIES)
    unit = random.choice(["C","F"])
    tmpl = random.choice(WEATHER_TEMPLATES)
    if "{unit_word}" in tmpl or "{unit_letter}" in tmpl:
        word = random.choice(WEATHER_UNIT_WORDS[unit])
        prompt = tmpl.format(city=city, unit_word=word, unit_letter=unit)
    else:
        prompt = tmpl.format(city=city)
    return {
        "history": [], "prompt": prompt,
        "expected": {"type":"tool_call","tool":"weather","args":{"location":city,"unit":unit}},
        "response": mk_tool_call("weather",{"location":city,"unit":unit}),
    }

def gen_calendar_list():
    d = random_date()
    prompt = random.choice(CAL_LIST_TEMPLATES).format(date=d)
    return {
        "history": [], "prompt": prompt,
        "expected": {"type":"tool_call","tool":"calendar","args":{"action":"list","date":d}},
        "response": mk_tool_call("calendar",{"action":"list","date":d}),
    }

def gen_calendar_create():
    d     = random_date()
    title = random.choice(EVENT_TITLES)
    prompt = random.choice(CAL_CREATE_TEMPLATES).format(date=d, title=title)
    return {
        "history": [], "prompt": prompt,
        "expected": {"type":"tool_call","tool":"calendar","args":{"action":"create","date":d,"title":title}},
        "response": mk_tool_call("calendar",{"action":"create","date":d,"title":title}),
    }

def gen_convert():
    from_u, to_u = random.choice(ALL_UNIT_PAIRS)
    value  = random_amount(0.001, 10000)
    prompt = random.choice(CONVERT_TEMPLATES).format(
        value=value, from_unit=from_u, to_unit=to_u)
    return {
        "history": [], "prompt": prompt,
        "expected": {"type":"tool_call","tool":"convert","args":{"value":value,"from_unit":from_u,"to_unit":to_u}},
        "response": mk_tool_call("convert",{"value":value,"from_unit":from_u,"to_unit":to_u}),
    }

def gen_currency():
    from_, to = random.choice(CURRENCY_PAIRS)
    amount    = random_amount(1, 50000)
    from_name = CURRENCY_NAMES.get(from_, from_)
    to_name   = CURRENCY_NAMES.get(to, to)
    prompt = random.choice(CURRENCY_TEMPLATES).format(
        amount=amount, from_=from_, to=to,
        from_name=from_name, to_name=to_name)
    return {
        "history": [], "prompt": prompt,
        "expected": {"type":"tool_call","tool":"currency","args":{"amount":amount,"from":from_,"to":to}},
        "response": mk_tool_call("currency",{"amount":amount,"from":from_,"to":to}),
    }

def gen_sql():
    query  = random_sql()
    prompt = random.choice(SQL_PROMPT_TEMPLATES).format(query=query)
    return {
        "history": [], "prompt": prompt,
        "expected": {"type":"tool_call","tool":"sql","args":{"query":query}},
        "response": mk_tool_call("sql",{"query":query}),
    }

def gen_refusal_chitchat():
    return {
        "history": [], "prompt": random.choice(CHITCHAT_PROMPTS),
        "expected": {"type":"refusal"},
        "response": random.choice(REFUSAL_RESPONSES),
    }

def gen_refusal_missing_tool():
    return {
        "history": [], "prompt": random.choice(MISSING_TOOL_PROMPTS),
        "expected": {"type":"refusal"},
        "response": random.choice(REFUSAL_RESPONSES),
    }

def gen_refusal_ambiguous():
    return {
        "history": [], "prompt": random.choice(AMBIGUOUS_PROMPTS),
        "expected": {"type":"refusal"},
        "response": random.choice(AMBIGUOUS_RESPONSES),
    }

def gen_multiturn_currency():
    from1, to1 = random.choice(CURRENCY_PAIRS)
    to2    = random.choice([c for c in CURRENCY_NAMES if c not in (from1, to1)])
    amount = random_amount(10, 5000)
    history = [
        {"role":"user","content":f"Convert {amount} {from1} to {to1}"},
        {"role":"assistant","content":mk_tool_call("currency",{"amount":amount,"from":from1,"to":to1})},
    ]
    prompt = random.choice([f"Now convert that to {to2}", f"And in {to2}?", f"What about {to2}?"])
    return {
        "history": history, "prompt": prompt,
        "expected": {"type":"tool_call","tool":"currency","args":{"amount":amount,"from":from1,"to":to2}},
        "response": mk_tool_call("currency",{"amount":amount,"from":from1,"to":to2}),
    }

def gen_multiturn_convert():
    from_u, to_u = random.choice(ALL_UNIT_PAIRS)
    v1 = random_amount(1, 1000)
    v2 = random_amount(1, 1000)
    history = [
        {"role":"user","content":f"Convert {v1} {from_u} to {to_u}"},
        {"role":"assistant","content":mk_tool_call("convert",{"value":v1,"from_unit":from_u,"to_unit":to_u})},
    ]
    return {
        "history": history, "prompt": f"What about {v2} {from_u}?",
        "expected": {"type":"tool_call","tool":"convert","args":{"value":v2,"from_unit":from_u,"to_unit":to_u}},
        "response": mk_tool_call("convert",{"value":v2,"from_unit":from_u,"to_unit":to_u}),
    }

def gen_adversarial():
    gen = random.choice([gen_weather, gen_currency, gen_convert])
    ex  = gen()
    if random.random() < 0.5:
        ex["prompt"] = add_typos(ex["prompt"])
    else:
        prefix = random.choice(CODE_SWITCH_PREFIXES)
        ex["prompt"] = prefix + ex["prompt"][0].lower() + ex["prompt"][1:]
    return ex


# def gen_adversarial():
#     gen = random.choice([gen_weather, gen_currency, gen_convert])
#     ex  = gen()
#     if random.random() < 0.5:
#         ex["prompt"] = add_typos(ex["prompt"])
#     else:
#         prefix = random.choice(CODE_SWITCH_PREFIXES)
#         ex["prompt"] = prefix + ex["prompt"][0].lower() + ex["prompt"][1:]
#     return ex

def gen_hindi_urdu_weather():
    city     = random.choice(CITIES)
    template, unit = random.choice(HINDI_URDU_WEATHER)
    prompt   = template.format(city=city)
    return {
        "history": [],
        "prompt": prompt,
        "expected": {
            "type": "tool_call",
            "tool": "weather",
            "args": {"location": city, "unit": unit}
        },
        "response": mk_tool_call("weather", {"location": city, "unit": unit}),
    }

GENERATORS = [
    (gen_weather,             0.20),
    (gen_calendar_list,       0.09),
    (gen_calendar_create,     0.09),
    (gen_convert,             0.15),
    (gen_currency,            0.13),
    (gen_sql,                 0.08),
    (gen_refusal_chitchat,    0.07),
    (gen_refusal_missing_tool,0.05),
    (gen_refusal_ambiguous,   0.05),
    (gen_multiturn_currency,  0.04),
    (gen_multiturn_convert,   0.03),
    (gen_adversarial,         0.02),
]

def weighted_choice():
    gens, weights = zip(*GENERATORS)
    return random.choices(gens, weights=weights, k=1)[0]

def generate_dataset(n: int) -> list:
    examples  = []
    seen      = set()
    attempts  = 0
    while len(examples) < n and attempts < n * 10:
        attempts += 1
        ex = weighted_choice()()
        h  = hashlib.sha256(ex["prompt"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        ex["id"] = f"synth_{len(examples):05d}"
        examples.append(ex)
    return examples

def to_chatml(ex: dict, system: str) -> dict:
    messages = [{"role": "system", "content": system}]
    for h in ex.get("history", []):
        messages.append(h)
    messages.append({"role": "user",      "content": ex["prompt"]})
    messages.append({"role": "assistant", "content": ex["response"]})
    return {"messages": messages, "id": ex["id"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=2000)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Generating {args.n} examples ...")
    examples = generate_dataset(args.n)
    random.shuffle(examples)

    split     = int(len(examples) * 0.95)
    train_ex  = examples[:split]
    val_ex    = examples[split:]

    with open(out_dir / "train.jsonl", "w") as f:
        for e in train_ex:
            f.write(json.dumps(to_chatml(e, SYSTEM_PROMPT), ensure_ascii=False) + "\n")

    with open(out_dir / "val.jsonl", "w") as f:
        for e in val_ex:
            f.write(json.dumps(to_chatml(e, SYSTEM_PROMPT), ensure_ascii=False) + "\n")

    hashes = {hashlib.sha256(e["prompt"].encode()).hexdigest(): e["id"] for e in examples}
    with open(out_dir / "prompt_hashes.json", "w") as f:
        json.dump(hashes, f)

    print(f"Done! Train: {len(train_ex)}  Val: {len(val_ex)}")

if __name__ == "__main__":
    main()