import json
from seg import segment_word

def segment_patterns_in_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for intent in data['intents']:
        if 'patterns' in intent:
            segmented_patterns = [segment_word(pattern) for pattern in intent['patterns']]
            intent['patterns'] = segmented_patterns

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_json_file = 'UniChatBot_segmented.json'
    output_json_file = 'UniChatBot_segmented.json'
    segment_patterns_in_json(input_json_file, output_json_file)
