#imports
from flask import Flask, render_template, request
from transformers import AutoTokenizer
import transformers

lmodel = "Felladrin/Pythia-31M-Chat-v1"
ltokenizer = AutoTokenizer.from_pretrained(lmodel)

pipeline = transformers.pipeline(
    "text-generation",
    model = lmodel,
    tokenizer = ltokenizer,
)


# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__)
#create chatbot
# englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = ChatterBotCorpusTrainer(englishBot)
# trainer.train("chatterbot.corpus.english") #train the chatter bot for english



#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    messageText = request.args.get('messageInput')
    sequences = pipeline(
        text_inputs=messageText,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=ltokenizer.eos_token_id,
    )
    return str(sequences)

if __name__ == "__main__":
    app.run()