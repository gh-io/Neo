from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process():
    data = request.get_json()
    prompt = data.get('prompt', '')
    result = {
        "response": f"Brain AI received: {prompt}",
        "insight": "Processed successfully by Python core"
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
