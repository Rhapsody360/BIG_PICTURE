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
app = Flask(__name__)

#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    messageText = request.args.get('msg')

    # temporary
    print("messageText:",messageText)
    # print("messageText:",messageText)
    # return str(messageText)
    # end temporary
    sequences = pipeline(
        text_inputs=messageText,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=ltokenizer.eos_token_id,
    )
    print("sequencing done")
    return str(sequences)

if __name__ == "__main__":
    app.run()