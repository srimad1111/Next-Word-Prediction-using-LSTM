A deep learning model trained for 50 epochs to predict the next word in a text sequence. The training dataset used is Shakespeare’s Hamlet, and the model is implemented using an LSTM RNN.

Main Files
hamlet.txt — training text corpus
LSTM.RNN — Jupyter notebook for training
next_word_lstm.h5 — trained model
next_word_lstm_model_with_early_stopping.h5 — improved model
tokenizer.pickle — saved tokenizer for inference
requirements.txt — packages used

Model Information
Architecture: Embedding → LSTM → Dense
Training Epochs: 50
Optimizer: Adam
Loss Function: Categorical Crossentropy

How to Use
Load the tokenizer (tokenizer.pickle)
Load the model (next_word_lstm.h5)
Input your text
Get predicted next word
Example (pseudo-code): model.predict(sequence)

Dependencies
Install using:
pip install -r requirements.txt

Deployed on railway
