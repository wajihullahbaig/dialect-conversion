
from pipline import DialectTranslationPipeline
from transformers import Seq2SeqTrainingArguments


pipeline = DialectTranslationPipeline()

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_total_limit=4,
    weight_decay=0.01,  # L2 regularization
)



# Run full pipeline
trainer = pipeline.run("Dataset.xlsx", training_args)

# Evaluate custom input
input_texts = [
    "I CoLoUr 🎨 the centre of my favourite book.",
    "The aeroplane ✈️ was delayed.",
    "They play football ⚽ every weekend.",
    "The football match.⚽ ",
    "Autumn 🍂 is my favorite season.",
    "She drives a lorry 🚚 for a living."

]
predictions = pipeline.predict([pipeline.preprocess_text(text) for text in input_texts])
for text, pred in zip(input_texts, predictions):
   print(f"Input: {text}\nOutput: {pred}\n")


 # Initialize pipeline for inference (loading from checkpoint)
inference_pipeline = DialectTranslationPipeline(
    model_name="t5-base",
    checkpoint_dir="results/checkpoint-117/"
)

# Generate predictions
input_texts = [
    "I CoLoUr 🎨 the centre of my favourite book.",
    "He is travelling ✈️ to the THEATRE.",
]
predictions = inference_pipeline.predict([pipeline.preprocess_text(text) for text in input_texts])

for input_text, pred in zip(input_texts, predictions):
    print(f"Input: {input_text}\nOutput: {pred}\n")


