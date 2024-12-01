from flask import Flask, render_template, request, jsonify
from chatbot import *
from flask_cors import CORS  # Import CORS
bot = chatbotAI()
app = Flask(__name__)
CORS(app, origins='http://localhost:3000',supports_credentials=True)  # Enable CORS for all routes

@app.route("/")
def index():
    return render_template("chat.html")
@app.route("/get/<string:msg>", methods=["GET","POST"])
def chat(msg):
    print(msg)
    response = bot.chatAPI(msg)
    return jsonify(response)

@app.route("/chatbot", methods=["GET","POST"])
def chatbot():
    msg = request.form.get('msg')
    response = bot.chatAPI(msg)
    return response
if __name__ == '__main__':
    app.run(port=5000, debug=True)
