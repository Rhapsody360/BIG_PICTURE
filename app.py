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
    start_keyword = """
    <|im_start|>system
    You are a very intelligent assistant. Help the user as much as you can.<|im_end|>
    <|im_start|>user
    """
    text_input = start_keyword +messageText+""" <|im_end|>
    <|im_start|>assistant
    """
    sequences = pipeline(
        text_inputs=text_input,
        max_length=500,
        truncation=True,
        do_sample=True,
        penalty_alpha= 0.5,
        top_k=2,
        repetition_penalty= 1.0016,
        num_return_sequences=1,
        eos_token_id=ltokenizer.eos_token_id,
    )
    print("sequencing done")

    output = str(sequences)
    intruncont = len(start_keyword) + len(messageText) + 66
    output = output[intruncont:-13]

    return output

if __name__ == "__main__":
    app.run()