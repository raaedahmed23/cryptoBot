# Seq2Seq Chatbot for Cryptocurrency
Incomplete. Need to add sample chats and upload model checkpoint file. 

## Scope 
The goal of the project was to develop a functioning chatbot using the seq2seq architecture to understand its mechanics and applications in the real-world.

This chatbot was designed to handle basic conversational talks such as greeting users, answering simple queries, and remembering the user's name and likes/dislikes. 

I used the tutorial available in the PyTorch documentation to create and train the model https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

Implemented using PyTorch version 2.1.2

## Model Architure
### Overview
The chatbot utilizes a sequence-to-sequence (seq2seq) model architecture, a common framework for building chatbots that can generate natural language responses. This model is composed of two main components: an encoder and a decoder, both of which are critical for processing and generating language.

The model training was done with the following parameters: 
hidden_size = 500\
encoder_n_layers = 4\
decoder_n_layers = 4\
dropout = 0.1\
batch_size = 4

### Encoder
The encoder in our model is a recurrent neural network (RNN) that processes the input sentence one word at a time. It converts each word into a hidden state and an output. The final hidden state of the encoder, which encapsulates the information of the entire input sentence, is then passed to the decoder to generate the corresponding output sentence. This model uses a gated recurrent unit (GRU) as the core of the encoder network, chosen for its efficiency and performance in capturing dependencies in the input sequence without the complexity of models like LSTM.

## Decoder
The decoder is also a GRU-based RNN that takes the final hidden state of the encoder as its initial hidden state. It begins generating the output sentence by predicting one word at a time. The decoder uses a technique called "teacher forcing" during training, where the target word at each time step is provided as the next input to the decoder, instead of using the decoder’s own previous prediction. This method helps speed up convergence and improves the training efficiency of the model.

### Attention Mechanism
An attention mechanism is integrated within the decoder. Attention allows the decoder to focus on different parts of the encoder’s outputs for each step of the output generation. This is particularly useful for longer input sequences, where the relevance of each word in the context of the conversation can vary significantly. The attention mechanism computes a context vector by taking a weighted sum of the encoder outputs, with weights determined by the decoder’s current state and each encoder output.

### Training
Computational Resources: Trained using a Kaggle notebook, which provided access to a GPU environment. Utilizing GPU resources was essential for handling the computational demands of training a seq2seq model.

The seq2seq model was trained with a batch size of 4, which was chosen to optimize the use of the GPU memory while allowing for sufficient gradient updates per epoch. The training process spanned over 100 epochs to ensure that the model had adequate exposure to the training data.

The model was trained using a cross-entropy loss function.

## Performance
### Current Capabilities
Post-training, the chatbot showed significant improvements in generating responses compared to its initial state, where it was outputting random words. This progress indicates that the model has begun to understand the structure and context of the conversations based on the training data provided.

### Limitations and Future Improvements
Despite the improvements, the chatbot still requires a more extensive dataset to effectively answer questions on specialized topics such as cryptocurrency. The current dataset, while useful for initial training, does not fully encompass the complexity and variety of language used in real-world cryptocurrency discussions. Expanding the dataset will likely help the model improve its accuracy and relevance in responses, making it more practical for seeking information on these topics.

## Sample Chats

Needs to be updated. 






