import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model():
    model_id = "RWKV/rwkv-4-169m-pile"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def generate_story(characters, age, moral):
    model, tokenizer = load_model()
    question = f"Write a story about {characters} for an audience aged {age} with the moral {moral}"
    prompt = f"### Instruction: {question}\n### Response:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(inputs["input_ids"], max_length=500, do_sample=True, temperature=0.7)

    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title('Story Writer App')
characters = st.text_input("Enter the characters for the story")
age = st.slider("Select the age of the audience", 5, 18, 10)
moral = st.text_input("Enter the moral of the story")

if st.button('Generate Story'):
    story = generate_story(characters, age, moral)
    st.text(story)
