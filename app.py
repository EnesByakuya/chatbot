from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_bot(user_input, chat_history_ids=None):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response, _ = chat_with_bot(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)