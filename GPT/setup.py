#Load model directly
from transformers import AutoTokenizer
import transformers


lmodel = "Felladrin/Pythia-31M-Chat-v1"
ltokenizer = AutoTokenizer.from_pretrained(lmodel)

pipeline = transformers.pipeline(
    "text-generation",
    model = lmodel,
    tokenizer = ltokenizer,
)

#Prompt the model with the required parameters
sequences = pipeline(
    text_inputs="i am depressed, i want to die somewhere",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=ltokenizer.eos_token_id,
)

#Output the inference
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

#sequences now contains the model output as a string and now you are required to add it to the html


# from transformers import pipeline

# model_name_or_path = "Amod/falcon7b-fine-tuned-therapy-merged" #path/to/your/model/or/name/on/hub
# pipe = pipeline("text-generation", model=model_name_or_path,trust_remote_code=True)
# print(pipe("i am really depressed"))

# from transformers import pipeline

# model_name_or_path = "kashif/stack-llama-2" #path/to/your/model/or/name/on/hub
# pipe = pipeline("text-generation", model=model_name_or_path,trust_remote_code=True)
# print(pipe("This movie was really")[0]["generated_text"])