"""
Step 2: turn a messy speech-to-text transcript into the sentence the user meant.

Run it on its own to test the prompt:   python3 interpret.py
Later, stt_test.py will call interpret() after the user pauses.

Setup:
    python3 -m pip install anthropic
    add ANTHROPIC_API_KEY=... to your .env file
"""

import json
import re

from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY from .env
client = anthropic.Anthropic()

MODEL = "claude-haiku-5-5"  # fast model, because the user is waiting on this

SYSTEM_PROMPT = """You are the interpretation layer of an assistive communication device \
for a person with dysarthria (slurred, slow, or imprecise speech).

A speech-to-text system listened to the person and produced a raw transcript. \
Because the speech is unclear, the transcript often contains wrong words, dropped \
consonants, split or merged words, or nonsense that sounds like what they said.

Your job: work out the sentence the person most likely MEANT to say.

How to do it:
- Read the raw transcript as sounds, not as spelling. Ask what real phrase it sounds like.
- Use the situation and the person's usual phrases as strong hints, but do not force a \
match if the transcript clearly says something else.
- Never add words or details the person did not say. No "please", no "a cup of", no \
extra items. Match the length of what they said.
- Keep it short and natural, in the first person, the way the person would speak.
- Do not correct their grammar or make them sound more formal than they are.
- Alternates must be different from "best" and from each other. They must be real, \
plausible sentences, never a garbled copy of the transcript.
- The usual phrases are hints only. Do not pick one unless the transcript actually \
sounds like it.
- If the transcript is just filler or noise ("uh", "mm", a single unclear sound) or you \
cannot tell what was meant, return "best" as an empty string "", alternates as [], and \
confidence "low". Never invent a sentence.
- If the person previously rejected a guess, do NOT return that guess again.
- The transcript may contain filler the speech-to-text invented ("okay", "whatever", \
   "oh man"). Ignore filler that does not fit the rest of what was said.

Respond with JSON only, no other text, in exactly this shape:
{"best": "<most likely sentence>", "alternates": ["<second guess>", "<third guess>"], \
"confidence": "high" | "medium" | "low"}"""


def build_user_message(raw, situation, phrases, confirmed, rejected):
    parts = [f'Raw transcript: "{raw}"']
    if situation:
        parts.append(f"Situation: {situation}")
    if phrases:
        parts.append("Phrases this person often says:\n- " + "\n- ".join(phrases))
    if confirmed:
        parts.append(
            "Recent sentences the person confirmed were correct:\n- "
            + "\n- ".join(confirmed[-5:])
        )
    if rejected:
        parts.append(
            "The person said these guesses were WRONG for this utterance:\n- "
            + "\n- ".join(rejected)
        )
    return "\n\n".join(parts)


def parse_json(text):
    """The model should return pure JSON, but be forgiving if it adds extra text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def interpret(raw, situation="", phrases=None, confirmed=None, rejected=None):
    """
    raw        -- the raw STT transcript
    situation  -- e.g. "ordering coffee at a cafe"
    phrases    -- list of phrases this person says a lot
    confirmed  -- list of sentences the person accepted earlier
    rejected   -- guesses the person already pressed WRONG on for this utterance

    Returns {"best": str, "alternates": [str, ...], "confidence": str}
    """
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_user_message(
                        raw, situation, phrases, confirmed, rejected
                    ),
                }
            ],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        result = parse_json(text)
        result.setdefault("alternates", [])
        result.setdefault("confidence", "medium")
        return result
    except Exception as e:
        # Never leave the user with a blank screen: fall back to the raw text.
        print("interpret() failed:", e)
        return {"best": raw, "alternates": [], "confidence": "low"}


# ---------------------------------------------------------------------------
# Test bench. Add your own cases here as you find new failures.
# Get realistic "raw" strings by running stt_test.py while a teammate imitates
# slurred or slow speech, then paste what Cartesia actually returned.
# ---------------------------------------------------------------------------
TEST_CASES = [
    # (raw transcript, situation)
    ("I wan a cah fee wih mil", "ordering coffee at a cafe"),
    ("can I have a lot day", "ordering coffee at a cafe"),
    ("ta oo ver mush", "ordering coffee at a cafe"),
    ("I nee to go to da bat room", "at a restaurant"),
    ("how mush is dat", "ordering coffee at a cafe"),
    ("uh", "ordering coffee at a cafe"),
    ("I will units.", "casual conversation"),  # meant: I really love unit 3
       ("Can I go? large oat milk latte. Okay.", "ordering coffee at a cafe"),  # meant: Can I get a large oat milk latte
   ("I'd. Like, Ike, a a chocolate.", "ordering coffee at a cafe"),  # meant: I'd like a small hot chocolate
   ("But where? is the bathroom.", "ordering coffee at a cafe"),  # meant: Where is the bathroom
   ("I'll pay with my my card.", "ordering coffee at a cafe"),  # meant: I'll pay with my card
   ("Can I get a medium iced coffee with X2? extra ice. And.", "ordering coffee at a cafe"),  # meant: Can I get a medium iced coffee with extra ice
]

PHRASES = [
    "Can I get a large oat milk latte",
    "Thank you very much",
    "How much is that",
    "Where is the bathroom",
]

if __name__ == "__main__":
    for raw, situation in TEST_CASES:
        out = interpret(raw, situation=situation, phrases=PHRASES)
        print(f'RAW:   {raw}')
        print(f'BEST:  {out["best"]}   ({out["confidence"]})')
        print(f'ALTS:  {out["alternates"]}\n')

    # Wrong-button test: pretend the user rejected the first guess.
    print("--- after pressing WRONG ---")
    first = interpret("can I have a lot day", situation="ordering coffee at a cafe")
    second = interpret(
        "can I have a lot day",
        situation="ordering coffee at a cafe",
        rejected=[first["best"]],
    )
    print("first guess: ", first["best"])
    print("second guess:", second["best"])