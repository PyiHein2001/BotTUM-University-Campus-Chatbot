import json
import random
from fuzzywuzzy import fuzz, process

# Load your JSON data
with open('test.json', 'r') as f:
    data = json.load(f)

def get_best_match(patterns, user_message):
    # Find the best match and its score using fuzzy matching
    best_match, score = process.extractOne(user_message, patterns, scorer=fuzz.token_sort_ratio)
    return best_match, score

def get_response(intents, user_message):
    threshold = 70  # Threshold for a good match (adjust as needed)
    for intent in intents:
        patterns = intent["patterns"]
        best_match, score = get_best_match(patterns, user_message)
        if score > threshold:
            return random.choice(intent["responses"])
    return "I'm not sure about that. Could you please provide more details?"

def main():
    print("Chatbot: Hello! How can I assist you today?")
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit"]:
            farewell_responses = [intent["responses"] for intent in data["intents"] if intent["tag"] == "farewell"]
            if farewell_responses:
                print("Chatbot:", random.choice(farewell_responses[0]))
            break
        response = get_response(data["intents"], user_message)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
