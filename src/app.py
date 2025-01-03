from flask import Flask, request, jsonify
from flask_cors import CORS
import os
print(os.getcwd())
from Painting.generation import generation

app = Flask(__name__)
CORS(app)  # Autorise les requêtes provenant de localhost pour le développement

@app.route('/process-prompts', methods=['POST'])
def process_prompts():
    data = request.json
    positive_prompt = data.get('positive_prompt', '')
    negative_prompt = data.get('negative_prompt', '')

    # Logique à effectuer avec les prompts
    print("Prompts received")
    image = generation(positive_prompt, negative_prompt)
    response = {
        "message": "Image generated",
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)