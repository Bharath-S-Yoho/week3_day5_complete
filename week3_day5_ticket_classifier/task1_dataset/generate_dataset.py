import csv
import random
from pathlib import Path

random.seed(42)
TEMPLATES = {
    "Billing": ["I was charged twice for my order", "Please explain this unexpected invoice",
                "My refund has not arrived", "The payment failed but money was deducted",
                "I need help with a billing error", "There is an incorrect charge on my account"],
    "Technical": ["The app crashes when I open it", "I cannot log in to my account",
                  "The website keeps showing an error", "My device will not sync",
                  "The page freezes after I click submit", "I need help fixing a technical issue"],
    "Delivery": ["My package has not arrived", "The tracking status has not changed",
                 "My order was delivered to the wrong address", "The shipment is delayed",
                 "Please update my delivery date", "The parcel arrived damaged"],
    "Account": ["I want to change my account email", "Please help me reset my password",
                "My account has been locked", "I cannot update my profile",
                "I need to close my account", "I cannot verify my account"],
}
PRIORITY_PHRASES = {
    "High": ["urgent", "as soon as possible", "blocking my work", "immediately"],
    "Medium": ["please help", "I need assistance", "please check"],
    "Low": ["just asking", "no rush", "for information", "when you have time"],
}

def main():
    output = Path(__file__).with_name("tickets.csv")
    rows = []
    for category, phrases in TEMPLATES.items():
        for ticket_number in range(150):
            priority = random.choice(list(PRIORITY_PHRASES))
            text = f"{random.choice(phrases)}. {random.choice(PRIORITY_PHRASES[priority])}. Reference {category.lower()} ticket {ticket_number + 1}."
            rows.append({"text": text, "category": category, "priority": priority})
    random.shuffle(rows)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "category", "priority"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} synthetic tickets at {output}")

if __name__ == "__main__":
    main()
