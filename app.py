!pip install streamlit
# Import necessary libraries
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set model id
model_id = "RWKV/rwkv-raven-1b5"

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def generate_story(characters, age, moral):
    # Combine the inputs into a prompt for the model
    prompt = f"### Instruction: A story about {characters}, for an audience around {age} years old, with a moral that {moral}.\n### Response:"

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(inputs["input_ids"], max_length=500, temperature=0.9)

    # Return the generated story
    return tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

# Define Streamlit structure
def main():
    st.title("Story Writer App")
    
    # Take inputs from the user
    characters = st.text_input("Enter the characters for the story")
    age = st.slider("Select the age of the audience", 1, 100, 25)
    moral = st.text_input("Enter the moral of the story")

    if st.button("Generate Story"):
        # Generate the story
        story = generate_story(characters, age, moral)

        # Display the generated story
        st.text(story)

if __name__ == '__main__':
    main()
