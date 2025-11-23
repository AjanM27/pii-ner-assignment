import jsonlines
import random
import re
from faker import Faker

fake = Faker()

# --- STT NOISE PATTERNS ---
FILLERS = ["uh", "um", "like", "you know", "ok so", "i mean", "basically"]
NOISE = ["static", "background noise", "unintelligible", "garbled"]

DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

def spoken_digits(n=16):
    return " ".join(DIGIT_WORDS[str(random.randint(0, 9))] for _ in range(n))

def maybe_insert_noise(text):
    if random.random() < 0.3:
        idx = random.randint(0, len(text.split()))
        noisy = text.split()
        noisy.insert(idx, random.choice(NOISE))
        return " ".join(noisy)
    return text

def maybe_add_typos(text):
    if random.random() < 0.15:
        pos = random.randint(0, len(text)-2)
        text = text[:pos] + text[pos+1] + text[pos] + text[pos+2:]
    return text

# --- ENTITY GENERATORS ---
def gen_credit_card():
    if random.random() < 0.5:
        return spoken_digits(16)
    else:
        digits = "".join(str(random.randint(0,9)) for _ in range(16))
        return " ".join(digits)

def gen_phone():
    if random.random() < 0.5:
        # spoken phone
        return spoken_digits(10)
    else:
        # numeric phone
        return " ".join(str(random.randint(0, 9)) for _ in range(10))

def gen_email():
    email = fake.email().lower()
    email = email.replace(".", " dot ").replace("@", " at ")
    return email

def gen_name():
    return fake.name().lower()

def gen_date():
    d = fake.date().replace("-", " ")
    return d

def gen_city():
    return fake.city().lower()

def gen_location():
    loc = fake.address().lower().replace("\n", " ")
    return loc

GEN = {
    "CREDIT_CARD": gen_credit_card,
    "PHONE": gen_phone,
    "EMAIL": gen_email,
    "PERSON_NAME": gen_name,
    "DATE": gen_date,
    "CITY": gen_city,
    "LOCATION": gen_location,
}

ALL = list(GEN.keys())

def add_fillers(text):
    if random.random() < 0.3:
        filler = random.choice(FILLERS)
        return f"{filler} {text}"
    return text

def generate_example(i):
    chunks = []
    entities = []

    num_entities = random.randint(1, 3)
    for _ in range(num_entities):
        label = random.choice(ALL)
        val = GEN[label]()

        # Apply noise
        val = maybe_add_typos(val)
        val = maybe_insert_noise(val)

        # Where does the entity land?
        start = len(" ".join(chunks))
        if start > 0:
            start += 1

        chunks.append(val)
        end = start + len(val)

        entities.append({"start": start, "end": end, "label": label})

    full = " ".join(chunks)
    full = add_fillers(full)

    return {"id": f"utt_{i}", "text": full, "entities": entities}

def write(path, n):
    with jsonlines.open(path, "w") as w:
        for i in range(n):
            w.write(generate_example(i))

if __name__ == "__main__":
    write("data/train.jsonl", 900)
    write("data/dev.jsonl", 160)
    write("data/test.jsonl", 120)
    print("Generated dataset.")
