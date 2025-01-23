import streamlit as st
import requests

# FastAPI backend URL 
backend_url = "http://127.0.0.1:8000/predict"

# Set up the title and input box for user interaction
st.title("Twitter Text Sentiment Analysis")

# Create an input text box for the user to enter text
user_input = st.text_area("Enter the text you'd like to analyze:", "")
       
# Button to trigger prediction
if st.button("Analyze Sentiment"):
    if user_input:
        # Send a POST request to the FastAPI backend with the user's input text
        response = requests.post(backend_url, json={"text": user_input})

        if response.status_code == 200:
            # Get the sentiment result from the response
            result = response.json()
            print(result)
            st.write(f"Text: {result['text']}")
            st.write(f"Predicted Sentiment: {result['sentiment']}")
        else:
            st.error("Error: Could not connect to the backend.")

    else:
        st.warning("Please enter some text.")
