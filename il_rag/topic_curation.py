"""Hand curation of the topic layer's keywords.

c-TF-IDF picks the terms most DISTINCTIVE to a cluster, which is not the same as
the terms that describe its theme. On this corpus it also surfaces text-
extraction debris ("ofthe", "trou" + "bleshooting"), disclaimer and syndication
boilerplate ("motley fool", "factbased news"), phrases torn across a boundary
("approach technical"), dates and dollar figures ("june 8th", "157 billion"),
filler words ("setting", "coming weeks") and bystanders' names. None of that is
a theme, and all of it flows into the topic labels, the cross-tab and the
keyword-retention scores.

Two layers, applied in load_topic_info() so every reader sees the same list:

  GLOBAL_DISCARD   words that are junk in ANY fit: extraction debris, boilerplate,
                   filler. Also applied at fit time, together with the
                   date/number rule, so a refit never re-admits them.
  TOPIC_DISCARD    per-topic decisions for ONE fit, keyed by topic id. Topic ids
                   are renumbered by every refit, so these apply only while
                   topic_info.json still carries CURATED_FITTED_AT. A word that
                   is junk in one topic ("state", "market", "class") can be the
                   theme of another, which is why these are not global.

Nothing is deleted: a curated record keeps `keywords_raw` and
`keywords_discarded` (each with its reason), so the removal is auditable.
"""
import re

GENERIC = "common word"
NONSENSE = "nonsense / broken token"
NUMBER = "date, number or dollar fragment"
BOILERPLATE = "boilerplate (disclaimer, byline, citation, filing header)"
FRAGMENT = "phrase fragment"
NAME = "incidental name"
OFFTOPIC = "unrelated to the topic"

_CODES = {"G": GENERIC, "N": NONSENSE, "D": NUMBER, "B": BOILERPLATE,
          "F": FRAGMENT, "P": NAME, "O": OFFTOPIC}

# The fit TOPIC_DISCARD was written against (topic_info.json "fitted_at").
CURATED_FITTED_AT = "2026-09-03T18:08:50"

_GLOBAL = {
    "N": [
        "ofthe", "ifthe", "dierent", "veried", "trou", "bleshooting", "wi",
        "plaus", "supp", "ga", "gf", "mi", "cent", "googls", "03mini",
        "constitutionjanuary 2026", "testreportfromitemandcall", "sysexit0",
        "conftestpy",
    ],
    "B": [
        "commentary", "guarantees", "accordingly", "warranties", "offered quality",
        "quality opinions", "shall construed", "authorbased", "motley fool",
        "gurufocus", "fool adisclosure", "adisclosure policy", "policy link",
        "newsrx", "factbased news", "reports deliver", "deliver factbased",
        "discoveries world", "news research", "world copyright",
        "research discoveries", "according news", "news past", "past week",
        "toi tech", "news toi", "tech news", "research markets", "content expressly",
        "post posthaven", "posthaven user", "upvoted", "states edition",
        "american daily", "daily stock", "report united", "cited page", "link pdf",
        "case number", "working paper", "conference paper", "published conference",
        "previous page", "continued previous", "page evaluating", "press releases",
        "anthropic said", "google says", "openai news", "positive sentiment",
        "negative sentiment", "stories impacting", "impacting alphabet",
        "stock forecast", "better alphabet", "like better", "stocks like",
    ],
    "G": [
        "set", "come", "setting", "prompted setting", "coming weeks", "cut",
        "criteria", "observed", "concerning", "overview", "sorry", "thank", "think",
        "organization", "supported", "new model", "condition", "assumption",
        "claim", "jumps", "reached", "tracked", "egregious", "tone", "overhaul",
        "surveyed", "fourth", "nominal", "owned", "decrease", "indicating",
        "lessons learned", "easy medium", "advised", "pretty wrong", "delighted",
        "necessity", "realized", "subset", "skew", "inquiries", "pointing",
        "help organizations", "help customers", "seamlessly", "avert", "stalled",
        "cached", "outdated", "existing users", "current systems",
        "todays systems", "ai advancements", "user base", "ai response",
        "response type", "public models",
    ],
}
GLOBAL_DISCARD = {kw: _CODES[c] for c, words in _GLOBAL.items() for kw in words}

# topic id -> reason code -> keywords. Global words are repeated here where they
# occur, so this table alone is the complete record for CURATED_FITTED_AT.
_TOPIC = {
    -1: {"G": ["access", "technology"]},
    0: {"G": ["works", "class"]},
    1: {"G": ["public", "private"]},
    2: {"B": ["link", "pages", "cited page"], "P": ["chen"],
        "F": ["association computational", "processing systems"]},
    3: {"G": ["setting", "cut", "criteria"]},
    4: {"G": ["observed", "concerning", "simulated"]},
    5: {"G": ["operation"], "B": ["anthropic said"]},
    6: {"G": ["state"], "F": ["general james"]},
    7: {"F": ["policy anthropic"]},
    8: {"G": ["overview", "sorry", "user", "assistant"], "N": ["ofthe", "ifthe"]},
    9: {"F": ["analysis insights", "insights training", "methods analysis",
              "models methods", "scaling language"],
        "O": ["high school", "common sense"]},
    10: {"P": ["mallaby", "demis"], "F": ["infinity"]},
    11: {"B": ["commentary", "guarantees", "accordingly", "warranties", "content",
               "stories", "offered quality", "quality opinions", "shall construed",
               "authorbased"]},
    12: {"G": ["letter"]},
    13: {"F": ["development information"], "G": ["centre", "programmes", "lab"]},
    14: {"N": ["constitutionjanuary 2026"], "F": ["want claude", "operator user"]},
    17: {"F": ["approach technical", "technical agi", "safety security"],
         "G": ["judges"]},
    18: {"G": ["new model", "cache"], "D": ["30 hours"]},
    19: {"G": ["chair", "thank", "think", "professor"]},
    20: {"G": ["organization"], "D": ["157 billion", "raising 66", "2024 raising",
                                      "valued 157"],
         "F": ["openai shapes", "democratization trends", "shapes ai"]},
    21: {"N": ["dierent"], "G": ["social"]},
    22: {"G": ["score"]},
    23: {"F": ["openai ceo", "ceo sam"], "D": ["june 01", "june 03", "2026 openai"],
         "G": ["announcements"], "B": ["openai news"]},
    25: {"G": ["severe"]},
    27: {"G": ["answers", "supported"], "F": ["teaching language", "models support"],
         "N": ["veried"]},
    28: {"F": ["35 sonnet", "upgraded claude"], "N": ["0shot", "5shot"],
         "D": ["64k", "200k"]},
    29: {"P": ["molo"]},
    30: {"G": ["prompted setting"]},
    31: {"G": ["weak"], "O": ["puzzles"]},
    32: {"G": ["countries"]},
    33: {"G": ["condition"], "F": ["models harmful", "evaluating language"]},
    34: {"N": ["cent"], "D": ["feb", "380 billion", "70 billion"], "O": ["menlo"],
         "G": ["designation"]},
    35: {"D": ["250 million"], "B": ["upvoted"]},
    36: {"G": ["chairman"]},
    37: {"N": ["trou", "bleshooting"], "F": ["red", "teamers"]},
    39: {"O": ["ucl"], "D": ["51 billion"], "G": ["londonbased"]},
    40: {"P": ["fleming", "modiano"], "F": ["founder experience"], "O": ["washes"]},
    41: {"B": ["news past", "past week", "weekly", "desk", "toi tech",
               "research markets", "tech news", "news toi"],
         "G": ["announcements"]},
    42: {"B": ["working paper"], "P": ["acemoglu", "autor"]},
    43: {"B": ["paper colm", "conference paper", "colm 2025", "published conference"],
         "G": ["ai response", "response type"]},
    44: {"B": ["google says"]},
    45: {"F": ["labs"]},
    46: {"G": ["regulated", "trusted"], "F": ["claude financial"]},
    47: {"G": ["assumption", "claim", "jumps"]},
    48: {"G": ["house"], "F": ["super"]},
    49: {"D": ["175b", "6b", "13b"], "F": ["public nlp"]},
    50: {"G": ["round"], "F": ["venture partners"],
         "D": ["615 billion", "183 billion", "13 billion"]},
    51: {"F": ["reasoning cup", "complex government", "data ai"], "D": ["june 2026"],
         "G": ["vehicle"]},
    52: {"G": ["projects", "lands"]},
    53: {"G": ["letter", "orders"]},
    54: {"G": ["studio", "anthropic models"], "F": ["word excel", "anthropics claude"]},
    55: {"G": ["messages", "sample", "panel", "plurality", "asking"]},
    56: {"D": ["30 billion", "10 billion", "invest 10"], "F": ["blackwell vera"]},
    57: {"P": ["lindsey", "seymour"], "G": ["injected", "evil"],
         "O": ["nuclear", "60 minutes"]},
    58: {"G": ["pro", "reached"], "F": ["25 deep"], "N": ["ga"]},
    59: {"G": ["tracked"], "F": ["risk severe", "framework beta"]},
    60: {"O": ["dealbook"], "P": ["dario", "sorkin", "cooper"],
         "F": ["anthropic ceo", "think players"]},
    61: {"G": ["current systems", "todays systems", "frozen"], "P": ["demis"],
         "F": ["artificial general"]},
    62: {"G": ["pro", "egregious", "tone"],
         "F": ["evaluation measuring", "compared gemini"]},
    63: {"D": ["tens billions"], "F": ["tensor", "processing units"]},
    64: {"G": ["specialization"], "P": ["hart"]},
    65: {"G": ["overhaul", "mobile", "coming weeks"],
         "F": ["chat dead", "tools ai"], "B": ["financial times"]},
    66: {"O": ["penn"], "G": ["usage", "using claude", "surveyed"]},
    67: {"G": ["nexus"], "P": ["gowers"]},
    68: {"F": ["integrated development", "development environment"]},
    69: {"O": ["einstein"], "P": ["kohli"], "F": ["deepmind ceo"]},
    70: {"G": ["quarter", "fourth"], "B": ["llc"],
         "F": ["shares information", "services providers", "providers stock",
               "information services", "alphabet worth"]},
    71: {"G": ["branches", "american people", "nominal"], "O": ["virtualitics"],
         "F": ["general services"]},
    72: {"O": ["computer program", "hole", "ukraine", "twinkle", "crows", "metres"],
         "N": ["plaus", "supp", "im"]},
    73: {"P": ["lee", "kim"], "G": ["campus"], "F": ["south"]},
    74: {"N": ["conftestpy", "sysexit0", "testreportfromitemandcall"],
         "G": ["return", "exit", "method"]},
    75: {"B": ["motley fool", "gurufocus", "content", "views", "warranties",
               "fool adisclosure", "adisclosure policy", "guarantees", "accordingly",
               "policy link"]},
    76: {"F": ["shares alphabet", "alphabet research"], "G": ["buy", "price"],
         "D": ["report thursday", "thursday april", "april 30th"],
         "B": ["research report"]},
    77: {"F": ["majority revenue", "advertisers consumers", "consumers worldwide",
               "connecting advertisers"],
         "B": ["better alphabet", "like better", "stocks like"]},
    78: {"F": ["free pro", "claude free", "pro max", "claude work"],
         "G": ["opt", "existing users"]},
    79: {"D": ["2025 available"]},
    80: {"D": ["monday june", "641 alphabet", "june 8th", "date monday",
               "200 billion"],
         "B": ["positive sentiment"], "O": ["mizuho"]},
    81: {"F": ["shares quarter", "providers stock", "services providers",
               "shares information", "information services", "stock valued",
               "acquiring additional", "holdings shares", "quarter finally"],
         "G": ["owns"]},
    82: {"F": ["shares company"], "G": ["sold", "average price", "owned", "decrease"]},
    83: {"G": ["ratio"], "F": ["day moving"],
         "D": ["billion quarter", "reported 511", "revenue 10990", "29th information",
               "equity 3899"]},
    84: {"D": ["200 million"], "G": ["global enterprises"],
         "F": ["snowflakes ai", "bring agentic"]},
    85: {"N": ["10x"], "F": ["claude life"], "G": ["spatial", "symposium"]},
    86: {"F": ["kings", "cross"], "O": ["dealroom"]},
    87: {"F": ["evaluations advanced", "safety responsibility"],
         "G": ["lessons learned"], "B": ["weidinger 2023"]},
    88: {"G": ["network", "challenges", "easy medium"],
         "F": ["ability discover", "models ability"]},
    89: {"P": ["henry"], "O": ["cambridge", "job market"],
         "F": ["professor university"]},
    90: {"B": ["press releases"], "F": ["round include"], "O": ["asr"],
         "D": ["20b series", "raises 20b"]},
    91: {"F": ["office ai", "grand"], "N": ["oai"], "G": ["council", "trusts"]},
    92: {"F": ["alleges anthropic"]},
    93: {"G": ["sites", "annotations"], "F": ["business enterprise",
                                            "analytics creative"]},
    94: {"G": ["lab", "new materials"]},
    95: {"F": ["hut", "bend", "river"], "N": ["mw"], "D": ["15year"]},
    96: {"F": ["creations", "online"], "O": ["ashurst"], "G": ["advised"]},
    97: {"G": ["deals"], "F": ["intelligence chips"]},
    98: {"G": ["pretty wrong", "delighted", "ai advancements"],
         "F": ["bank australia", "social economic"], "O": ["sydney"]},
    99: {"N": ["mi"], "G": ["market"], "B": ["breakdowns", "forecasts"]},
    100: {"G": ["necessity", "realized"], "F": ["grow ai"]},
    101: {"F": ["ai sas", "ai inside"], "G": ["cultural context"]},
    102: {"P": ["wulfmeier", "markus", "velloso"], "F": ["learning robotics"]},
    103: {"B": ["newsrx", "factbased news", "reports deliver", "deliver factbased",
                "discoveries world", "news research", "world copyright",
                "research discoveries", "according news"]},
    104: {"P": ["kwon", "ryu"], "G": ["governmental"]},
    105: {"F": ["anthropic business", "expansion partnership", "enterprises ai"],
          "G": ["reinvention", "help enterprises"]},
    106: {"N": ["wi"], "G": ["subset", "gold", "deliverable", "reference files"]},
    107: {"F": ["policy pos", "argument point", "pos argument"],
          "B": ["manipulation table", "page evaluating", "continued previous",
                "previous page"]},
    108: {"F": ["yvonne gonzalez", "judge yvonne", "president greg"]},
    109: {"G": ["indicating", "ratio"], "F": ["suggest stock", "strong financial"]},
    110: {"P": ["alexandr", "chan"], "F": ["machines lab"]},
    112: {"D": ["monday june", "june 8th", "billion quarter", "june 15th",
                "paid monday"],
          "G": ["quarterly"]},
    113: {"P": ["jeng"], "G": ["country"]},
    114: {"F": ["price work"], "G": ["skew", "inquiries"]},
    115: {"B": ["content expressly", "discretion", "sole", "reserve", "delete",
                "journal", "upvoted", "post posthaven", "posthaven user", "stories"]},
    116: {"F": ["magic", "asus dell", "dell hp"], "G": ["pointing"]},
    117: {"G": ["help organizations", "help customers"], "F": ["turn ai"]},
    118: {"F": ["mini", "instant", "enterprise edu"], "N": ["03mini"]},
    119: {"F": ["purpose work", "kelly head"], "G": ["connector", "seamlessly"]},
    120: {"D": ["12b", "31b", "256k context"], "N": ["e4b"], "O": ["hirundos"],
          "F": ["weight level"]},
    122: {"G": ["pilot"], "F": ["max plan", "sharing personal"]},
    123: {"G": ["size", "gold"]},
    124: {"D": ["80 million", "20 researchers"],
          "B": ["states edition", "american daily", "daily stock", "report united"],
          "F": ["class shares"]},
    125: {"G": ["sky", "client", "market insights"], "F": ["team led"]},
    126: {"B": ["negative sentiment", "news search"],
          "F": ["google moves", "deepmind train"], "G": ["avert"]},
    127: {"F": ["gross"], "B": ["stock forecast"]},
    128: {"O": ["ny tech", "tech week"]},
    129: {"P": ["dutta"]},
    130: {"G": ["stalled", "son"], "F": ["backed openai"]},
    131: {"G": ["mode", "cached", "feature", "external services"]},
    132: {"G": ["concept", "techniques", "behaviour"], "F": ["vs algorithm"]},
    133: {"G": ["saved", "outdated", "update"], "D": ["april 2024"]},
    134: {"B": ["positive sentiment", "stories impacting", "impacting alphabet"],
          "N": ["googls", "gf"], "D": ["q1", "200 billion"]},
    135: {"F": ["highly capable", "capable ai", "applications services"],
          "G": ["brakes", "layer"]},
    136: {"G": ["agreements"], "D": ["40 evaluations"],
          "F": ["standards innovation", "ai national", "understanding frontier"]},
    137: {"F": ["texas new"],
          "D": ["800 permanent", "jobs 2400", "50 billion", "2400 construction",
                "create 800"]},
    138: {"P": ["diaz"], "F": ["growing times", "learning build"], "D": ["apps 110"],
          "G": ["user base"]},
    139: {"F": ["gpt4o gpt41", "productive evaluation", "involves enormous",
                "openais gpt4o", "evaluation ideas"],
          "G": ["public models", "exercise"]},
    140: {"B": ["llp", "case number", "link pdf"],
          "F": ["pbc case", "anthropic represented", "court northern"],
          "D": ["number 324cv05417"], "P": ["joseph"], "O": ["lumina"]},
}
TOPIC_DISCARD = {
    t: {kw: _CODES[c] for c, words in codes.items() for kw in words}
    for t, codes in _TOPIC.items()
}

_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december", "jan", "feb", "mar",
           "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
             "sunday"}
_UNITS = {"billion", "billions", "million", "millions", "trillion", "thousand",
          "hundred", "q1", "q2", "q3", "q4"}
_NUMERIC = re.compile(r"^\d+(st|nd|rd|th|s|k|m|b|x|year|years)?$")


def is_date_or_number(keyword: str) -> bool:
    """True when EVERY token is a number, date word or money unit.

    "june 8th", "157 billion", "64k" go; "gpt4o", "claude 35", "asl3" stay,
    because a model or version name always carries a non-numeric token.
    """
    parts = str(keyword).lower().split()
    return bool(parts) and all(
        _NUMERIC.match(p) or p in _MONTHS or p in _WEEKDAYS or p in _UNITS
        for p in parts)


def discard_reason(keyword: str, topic: int | None = None,
                   fitted_at: str | None = None) -> str | None:
    """Why this keyword is not a theme word, or None if it stays."""
    if topic is not None and fitted_at == CURATED_FITTED_AT:
        reason = TOPIC_DISCARD.get(int(topic), {}).get(keyword)
        if reason:
            return reason
    if keyword in GLOBAL_DISCARD:
        return GLOBAL_DISCARD[keyword]
    if is_date_or_number(keyword):
        return NUMBER
    return None


def curate_keywords(keywords: list, topic: int | None = None,
                    fitted_at: str | None = None) -> tuple[list, list]:
    """(kept, discarded) — discarded as [{"keyword", "reason"}], order preserved."""
    kept, dropped = [], []
    for kw in keywords or []:
        reason = discard_reason(kw, topic, fitted_at)
        if reason:
            dropped.append({"keyword": kw, "reason": reason})
        else:
            kept.append(kw)
    return kept, dropped


def topic_label(topic: int, keywords: list, is_outlier: bool = False) -> str:
    if is_outlier:
        return "(unclustered)"
    if not keywords:
        # Topic id included so emptied topics stay distinct in charts keyed by
        # label.
        return f"(topic {topic}: no thematic keywords)"
    return ", ".join(keywords[:4])


def curate_info(info: dict) -> dict:
    """Apply curation to a loaded topic_info in place; idempotent."""
    if info is None or info.get("curated"):
        return info
    fitted_at = info.get("fitted_at")
    for r in info.get("topics", []):
        raw = r.get("keywords_raw", r.get("keywords", []))
        kept, dropped = curate_keywords(raw, r.get("topic"), fitted_at)
        r["keywords_raw"] = raw
        r["keywords"] = kept
        # A fit records what IT dropped; keep that alongside today's discards.
        r["keywords_discarded"] = r.get("keywords_discarded", []) + dropped
        r["label"] = topic_label(r.get("topic"), kept, r.get("is_outlier", False))
    info["curated"] = True
    return info
