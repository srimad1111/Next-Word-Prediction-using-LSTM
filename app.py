'''
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

    
'''

import dash
from dash import dcc, html, Input, Output
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None


# Build Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial", "maxWidth": "600px", "margin": "30px auto"},
    children=[
        html.H1("Next Word Prediction (LSTM)", style={"textAlign": "center"}),

        html.Label("Enter a sequence of words:"),
        dcc.Input(
            id='input-text',
            type='text',
            value="To be or not to",
            style={"width": "100%", "height": "40px", "padding": "10px"},
        ),

        html.Button(
            "Predict",
            id="predict-btn",
            n_clicks=0,
            style={"marginTop": "20px", "width": "100%", "height": "40px"},
        ),

        html.Div(id="output", style={"marginTop": "20px", "fontSize": "20px"})
    ]
)


@app.callback(
    Output("output", "children"),
    Input("predict-btn", "n_clicks"),
    Input("input-text", "value")
)
def update_output(n_clicks, input_text):
    if n_clicks == 0:
        return ""

    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

    return f"Next word: {next_word}" if next_word else "No prediction."


server = app.server  # for deployment

if __name__ == '__main__':
    app.run_server(debug=True)
