import pickle
import numpy as np

import string

from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer

from chatbot_model import voc, encoder, decoder, searcher, normalizeString, evaluate

wnl = WordNetLemmatizer()

print('Models built and ready to go!')


def welcome():
    print("*" * 40)
    print("Thanks for trying out the CryptoBot. Please keep in mind that this is an experimental Seq2Seq chatbot and may not always give a coherent output.")
    print("Important: To exit the chatbot you can type either 'exit', 'bye' or 'quit'")
    print("*" * 40, end="\n\n")

class CryptoChatBot:
    def __init__(self):
        self.users = pickle.load(open("users.pickle", "rb"))
        
        self.curr_user = None
        self.curr_user_info = {}
   
    def rizz_up(self):
        # asked for likes before calling this function 
        user_input = input("You: ")

        # extract named ent from input 
        likes = self.extract_topics(user_input)
        print(f"cryptoBot: I see. What about your dislikes?")
        
        user_input = input("You: ")
        dislikes = self.extract_topics(user_input)

        # store the likes and dislikes in the user_info 
        self.curr_user_info["likes"] = likes
        self.curr_user_info["dislikes"] = dislikes

        responses = ["You can ask me a crypto related question!", "Can I help you with a blockchain related query?", "How can I help you?"]
        print(f"cryptoBot: Thanks for your input. {np.random.choice(responses)}")

  
    def greet(self, query = None):
        greeting = ["Hi", "Hey", "Good day", "Hey there", "Greetings"]
        new_greet = ["Nice to meet you!", "Thanks for trying me out!"]
        return_greet = ["Good to have you back!", "Welcome back!", "Long time no see :)!"]
        
        if not query:
            return np.random.choice(greeting) + "! " + \
            "I'm a crypto chatbot designed to answer questions related to the world of cryptocurrency. What's your name?"  

        # do NER to find the name else most likely the last word of the response
        if self.curr_user is None:
            self.curr_user = self.find_name(query)

        # check is name exists in users dict -> returning user
        if self.curr_user in self.users:
            interest = self.users[self.curr_user]["likes"]
            response =  np.random.choice(greeting) + f" {self.curr_user}! " + np.random.choice(return_greet) \
                + f" Do you want to talk about {np.random.choice(interest)}?" 
            print(f"cryptoBot: {response}")
        else:
            response = np.random.choice(greeting) + f" {self.curr_user}! " + np.random.choice(new_greet) + \
                " Let's get to know you better. Could you tell me what your likes are?"
            
            print(f"cryptoBot: {response}")
            self.rizz_up()
            
       
    def bye(self):
        farewells = ["Bye! Sad to see you go :(", "See you next time!", "See ya! Have a good day"]
        # save new users info
        if self.curr_user not in self.users:
            self.users[self.curr_user] = self.curr_user_info
            # pickle the self.users dictionary
            pickle.dump(self.users, open("users.pickle", "wb"))

        return np.random.choice(farewells)
        

    def respond(self, query):
        confused = ["I don't relevant information regarding this. Anything else I can help you with?",\
                    "I'm not sure about that. Can I help with you something else?", \
                    "I can't assist you with that. Can I help you with something else?"]
        
        query_tokens = word_tokenize(query.lower())
        filtered_query = [word for word in query_tokens if word not in stopwords.words("english")]
        filtered_query = [word for word in filtered_query if word not in string.punctuation]
        
        if not filtered_query:
            return "Could you please provide more details in your question?"
        
        try: 
            input_sentence = normalizeString(query)
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            response = " ".join(output_words)
            return response

        except KeyError:
            return np.random.choice(confused)
        


    
    def extract_topics(self, query):
        tokens = word_tokenize(query)
        tagged_tokens = pos_tag(tokens)

        topics = [word for word, tag in tagged_tokens if "NN" in tag and word.lower() not in stopwords.words("english")]
        return topics


    def find_name(self, query):
        tokens = word_tokenize(query)
        pos_tags = pos_tag(tokens)
        named_ents = ne_chunk(pos_tags)
        name = [] 
        for chunk in named_ents:
            if hasattr(chunk, 'label') and chunk.label() == "PERSON":
                name.append(" ".join(c[0] for c in chunk))

        if not name:
            return tokens[-1]
        else:
            return name[0]

if __name__ == "__main__":
    chatbot = CryptoChatBot()

    welcome()
    print(f"cryptoBot: {chatbot.greet()}")

    name = False

    while True:
        user_input = input("You: ") 

        if user_input.lower() in ["bye", "exit", "quit"]:
            print(f"cryptoBot: {chatbot.bye()}")
            break
        if not name:
            chatbot.greet(user_input)
            name = True
        else:
            response = chatbot.respond(user_input)
            print(f"cryptoBot: {response}")
    


        