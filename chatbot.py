from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-large"  # Use a larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_bot(user_input, chat_history_ids=None):
    # Tokenize the user input
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)  # Create attention mask

    # Append the new input to the chat history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, inputs], dim=-1)
        attention_mask = torch.cat([torch.ones_like(chat_history_ids), attention_mask], dim=-1)
    else:
        bot_input_ids = inputs

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,  # Pass attention mask
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode and return the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

if __name__ == "__main__":
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")
    chat_history_ids = None
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response, chat_history_ids = chat_with_bot(user_input, chat_history_ids)
        print(f"Bot: {response}")