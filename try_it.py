from interpret import interpret

SITUATION = "ordering coffee at a cafe"
PHRASES = [
    "Can I get a large oat milk latte",
    "Thank you very much",
    "How much is that",
    "Where is the bathroom",
]
confirmed = []

while True:
    raw = input("\nraw transcript (blank to quit): ").strip()
    if not raw:
        break
    rejected = []
    while True:
        out = interpret(raw, situation=SITUATION, phrases=PHRASES,
                        confirmed=confirmed, rejected=rejected)
        if not out["best"]:
            print("-> couldn't tell, would ask the user to repeat")
            break
        print(f'-> {out["best"]}  ({out["confidence"]})  alts: {out["alternates"]}')
        key = input("Enter = accept, w = wrong: ").strip().lower()
        if key == "w":
            rejected.append(out["best"])
            continue
        confirmed.append(out["best"])
        print("accepted (this is where it would be spoken)")
        break