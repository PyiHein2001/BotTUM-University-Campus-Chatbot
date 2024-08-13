from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import json
import os
import uuid
import re
import random
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from collections import OrderedDict
from seg import segment_word

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management
CORS(app)

# Path to the JSON file for user credentials
CREDENTIALS_FILE = 'credentials.json'

# Path to the JSON file for user inputs
USER_INPUTS_FILE = 'user_inputs.json'

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the chatbot model, data, words, and classes
model = load_model('chatbot_model.h5')

# Compile the model if not already compiled
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    metrics=['accuracy']
)

with open('UniChatBot_segmented.json', encoding='utf-8') as file:
    data = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Define synonyms
synonyms = {
    "ကျောင်း": ["နည်းပညာ တက္ကသိုလ်", "ကျောင်း", "TUM", "နည်းပညာ တက္ကသိုလ် (မန္တလေး)", "တက္ကသိုလ်"],
    "လား": ["သလား"],
    "Civil Engineering": ["civil", "CIVIL", "Civil"],
    "Electronic Engineering": ["EC", "Ec", "ec"],
    "Electrical Power Engineering": ["EP", "Ep", "ep"],
    "Information Technology": ["IT", "It", "it", "အိုင်တီ"],
    "Mechanical Engineering": ["ME", "Me", "me"],
    "Mechatronic Engineering": ["MC", "Mc", "mc"],
}

def normalize_text(text, synonyms):
    """Replace synonyms in the text with the corresponding key."""
    for key, values in synonyms.items():
        for value in values:
            text = re.sub(rf'\b{re.escape(value)}\b', key, text, flags=re.IGNORECASE)
    return text

def myanmar_tokenize(text):
    """Tokenize Myanmar text with mixed English and Myanmar words."""
    segmented_text = segment_word(text)
    tokens = re.findall(r'[a-zA-Z]+|[\u1000-\u109F]+', segmented_text)
    return tokens

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence."""
    sentence_words = myanmar_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """Convert a sentence into a bag of words representation."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    """Predict the class of the sentence using the model."""
    segmented_sentence = myanmar_tokenize(sentence)
    segmented_sentence = ' '.join(segmented_sentence)
    normalized_sentence = normalize_text(segmented_sentence, synonyms)
    p = bow(normalized_sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    if results:
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    else:
        return_list.append({"intent": "noanswer", "probability": "1.0"})
    return return_list

def get_response(ints, intents_json):
    """Generate a response based on the predicted intent."""
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    for i in list_of_intents:
        if i['tag'] == "noanswer":
            return random.choice(i['responses'])
    return "မေးခွန်းတွက် သင့်တော်သော အဖြေ မပေးနိုင် ၍ ဝမ်းနည်း ပါတယ်။ ကျောင်းသားရေးရာ ဖုန်းနံပါတ် ၀၉-၉၈၈၄၈၄၁၇၂ ကို ဆက်သွယ်၍ အသေးစိတ်ကို မေးမြန်း နိုင်ပါတယ် ။"

def chatbot_response(text):
    """Generate a response from the chatbot."""
    ints = predict_class(text, model)
    res = get_response(ints, data)
    if not res:
        res = "မေးခွန်းတွက် သင့်တော်သော အဖြေ မပေးနိုင် ၍ ဝမ်းနည်း ပါတယ်။ ကျောင်းသားရေးရာ ဖုန်းနံပါတ် ၀၉-၉၈၈၄၈၄၁၇၂ ကို ဆက်သွယ်၍ အသေးစိတ်ကို မေးမြန်း နိုင်ပါတယ် ။"
    return res, ints[0]['intent']

# Load credentials from the JSON file
def load_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        return {'users': [{'username': 'admin', 'password': 'password', 'is_main': True}]}
    with open(CREDENTIALS_FILE, 'r') as f:
        return json.load(f)

# Save credentials to the JSON file
def save_credentials(credentials):
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials, f)

# Initialize the JSON file for user inputs
def initialize_json_file():
    if not os.path.exists(USER_INPUTS_FILE):
        with open(USER_INPUTS_FILE, 'w', encoding='utf-8') as file:
            json.dump({'user_inputs': []}, file, ensure_ascii=False, indent=4)

initialize_json_file()

def log_user_input(user_id, user_message, predicted_intent, feedback=None):
    """Log user inputs to a JSON file."""
    unique_id = str(uuid.uuid4())
    new_input = {
        "id": unique_id,
        "user_id": user_id,
        "message": user_message,
        "predicted_intent": predicted_intent,
        "feedback": feedback
    }
    with open(USER_INPUTS_FILE, 'r+', encoding='utf-8') as file:
        user_inputs = json.load(file)
        for entry in user_inputs['user_inputs']:
            if 'id' not in entry:
                entry['id'] = str(uuid.uuid4())
        user_inputs['user_inputs'].append(new_input)
        file.seek(0)
        json.dump(user_inputs, file, ensure_ascii=False, indent=4)
    return unique_id

# Route to serve the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to serve the admin page
@app.route('/admin.html')
def admin_page():
    if not session.get('authenticated'):
        return redirect(url_for('login_page'))
    return render_template('admin.html')

# Route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    bot_response, predicted_intent = chatbot_response(user_message)
    message_id = log_user_input("anonymous", user_message, predicted_intent)
    response = {
        "response": bot_response,
        "intent": predicted_intent,
        "message_id": message_id
    }
    return jsonify(response)

# Route to handle login
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    credentials = load_credentials()
    user = next((user for user in credentials['users'] if user['username'] == username and user['password'] == password), None)
    if user:
        session['authenticated'] = True
        session['username'] = username
        session['is_main'] = user['is_main']  # Store if the user is the main admin in the session
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid credentials'})

# Route to handle logout
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('authenticated', None)
    session.pop('username', None)
    session.pop('is_main', None)
    return jsonify({'success': True})

# Route to check authentication and return the username and main admin status
@app.route('/check_auth')
def check_auth():
    authenticated = session.get('authenticated')
    username = session.get('username') if authenticated else None
    is_main = session.get('is_main') if authenticated else None
    return jsonify({'authenticated': authenticated, 'username': username, 'is_main': is_main})

# Route to handle updating password
@app.route('/update_password', methods=['POST'])
def update_password():
    if not session.get('authenticated'):
        return jsonify({'success': False, 'message': 'Not authenticated'})
    data = request.json
    username = session.get('username')
    new_password = data.get('password')
    
    credentials = load_credentials()
    for user in credentials['users']:
        if user['username'] == username:
            user['password'] = new_password
            break
    save_credentials(credentials)
    return jsonify({'success': True})

# Route to add a new user
@app.route('/add_user', methods=['POST'])
def add_user():
    if not session.get('authenticated') or not session.get('is_main'):
        return jsonify({'success': False, 'message': 'Not authorized'})
    data = request.json
    new_user = {
        'username': data.get('username'),
        'password': data.get('password'),
        'is_main': False
    }
    
    credentials = load_credentials()
    credentials['users'].append(new_user)
    save_credentials(credentials)
    return jsonify({'success': True})

# Route to switch the main admin
@app.route('/switch_main_admin', methods=['POST'])
def switch_main_admin():
    if not session.get('authenticated') or not session.get('is_main'):
        return jsonify({'success': False, 'message': 'Not authorized'})
    data = request.json
    new_main_username = data.get('username')
    
    credentials = load_credentials()
    for user in credentials['users']:
        user['is_main'] = (user['username'] == new_main_username)
    save_credentials(credentials)
    return jsonify({'success': True})

# Route to handle user inputs API
@app.route('/api/user_inputs')
def api_user_inputs():
    with open(USER_INPUTS_FILE, 'r', encoding='utf-8') as file:
        user_inputs = json.load(file)
    return jsonify(user_inputs)

# Route to handle JSON data API for intents
@app.route('/api/json_data', methods=['GET', 'POST'])
def api_json_data():
    if request.method == 'POST':
        data = request.get_json()
        ordered_data = OrderedDict()
        ordered_data['intents'] = []
        for intent in data['intents']:
            ordered_intent = OrderedDict()
            ordered_intent['tag'] = intent['tag']
            ordered_intent['patterns'] = intent['patterns']
            ordered_intent['responses'] = intent['responses']
            ordered_intent['context'] = intent.get('context', [])
            ordered_data['intents'].append(ordered_intent)
        with open('UniChatBot_segmented.json', 'w', encoding='utf-8') as file:
            json.dump(ordered_data, file, ensure_ascii=False, indent=4)
        return jsonify({'status': 'success'})
    else:
        with open('UniChatBot_segmented.json', encoding='utf-8') as file:
            data = json.load(file, object_pairs_hook=OrderedDict)
        return jsonify(data)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Define the route for training the model
@app.route('/train_model', methods=['POST'])
def train_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.json'):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        training_logs, accuracy = train_model_from_file(file_path)
        return jsonify({"logs": training_logs, "accuracy": accuracy})

    return jsonify({"error": "Invalid file format"}), 400

def train_model_from_file(file_path):
    with open(file_path) as file:
        data = json.load(file)

    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

    lemmatizer = WordNetLemmatizer()

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '၊', '။']

    for intent in data['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array([item[0] for item in training])
    train_y = np.array([item[1] for item in training])

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Capture training logs
    class TrainingLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            self.logs = []

        def on_epoch_end(self, epoch, logs=None):
            self.logs.append(f"Epoch {epoch + 1}: loss = {logs['loss']:.4f}, accuracy = {logs['accuracy']:.4f}")

    logger = TrainingLogger()
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, callbacks=[logger])

    model.save('chatbot_model.h5')

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    accuracy = hist.history['accuracy'][-1] * 100
    return logger.logs, accuracy

@app.route('/segment_patterns', methods=['POST'])
def segment_patterns():
    data = request.json
    segmented_data = segment_patterns_in_data(data)
    return jsonify(segmented_data)

def segment_patterns_in_data(data):
    for intent in data['intents']:
        if 'patterns' in intent:
            segmented_patterns = [segment_word(pattern) for pattern in intent['patterns']]
            intent['patterns'] = segmented_patterns
    return data

# Route to serve the login page
@app.route('/login.html')
def login_page():
    return render_template('login.html')

# Route to serve the settings page
@app.route('/settings.html')
def settings_page():
    if not session.get('authenticated'):
        return redirect(url_for('login_page'))
    return render_template('settings.html')

# Route to handle feedback submission
@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_data = request.json
    feedback = feedback_data.get('feedback')
    message_id = feedback_data.get('id')

    with open(USER_INPUTS_FILE, 'r+', encoding='utf-8') as file:
        user_inputs = json.load(file)
        entry_found = False
        for entry in user_inputs['user_inputs']:
            if 'id' not in entry:
                print(f"Entry without id: {entry}")
            if entry['id'] == message_id:
                entry['feedback'] = feedback
                entry_found = True
                break
        if not entry_found:
            print(f"No entry found with id: {message_id}")
        file.seek(0)
        json.dump(user_inputs, file, ensure_ascii=False, indent=4)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
