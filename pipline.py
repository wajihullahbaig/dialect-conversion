from typing import Dict, List, Union
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer
from torch.utils.data import Dataset
import logging
import pandas as pd
import emoji
import re
import random
import torch
import nltk
nltk.download('wordnet',quiet=True)
from nltk.corpus import wordnet


class DialectTranslationPipeline:
    def __init__(self, 
                 model_name:str="t5-base", 
                 remove_emojis=True, 
                 deduplicate=True, 
                 augument_data = True,
                 tokenizer_name:str = "t5-base", 
                 log_file_path = "pipeline.log",
                 checkpoint_dir=None   
                 ):
        self.log_file_path = log_file_path
        self.configure_logger()
        random.seed(10)        
        try:
            self.model_name = model_name
            self.tokenizer_name = tokenizer_name
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.remove_emojis = remove_emojis
            self.deduplicate = deduplicate 
            self.augument_data = augument_data
             # Load model from checkpoint if provided
            if checkpoint_dir:
                self.checkpoint_dir = checkpoint_dir
                self.load_model_from_checkpoint(self.checkpoint_dir)
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {e}")
        

    def configure_logger(self):
        """Configure logging to both console and file handlers"""
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Configure console handler with high-level info
        self.logger.handlers.clear()
        self.logger.propagate = False
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Configure file handler with detailed debug info
        file_handler = logging.FileHandler(self.log_file_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def load_data(self, file_path:str) -> pd.DataFrame:
        """Load and parse Excel data with logging and sanity checks."""
        try:
            self.logger.info(f"Loading data from: {file_path}")
            excel_file = pd.ExcelFile(file_path)
            self.logger.debug(f"Found sheets: {excel_file.sheet_names}")
            df = excel_file.parse(excel_file.sheet_names[0])

            # Check for required columns
            required_columns = {'input_text', 'target_text'}
            if not required_columns.issubset(df.columns):
                self.logger.error(
                    "Error: Required columns 'input_text' and 'target_text' not found in Excel file."
                )
                raise ValueError(
                    "Excel file must contain 'input_text' and 'target_text' columns"
                )
            return df        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")

    def preprocess_dataframe(self, df:pd.DataFrame) -> pd.DataFrame:
        """Clean and deduplicate DataFrame with additional sanity checks."""
        self.logger.info("Preprocessing data")

        # Remove rows with missing or empty input/target texts
        df.dropna(subset=['input_text', 'target_text'], inplace=True)
        df = df[(df['input_text'].str.strip() != '') & (df['target_text'].str.strip() != '')]
        df.reset_index(drop=True, inplace=True)

        # Check if all rows were removed
        if df.empty:
            self.logger.error("No valid rows found after cleaning. Exiting.")
            raise ValueError("No valid data after cleaning")
        
        self.logger.info(f"Loaded {len(df.index)} valid rows from dataset")

        df['input_text'] = df['input_text'].apply(self.preprocess_text)
        df['target_text'] = df['target_text'].str.strip()

        # Remove empty rows created during preprocessing
        df = df[(df['input_text'].str.strip() != '') & (df['target_text'].str.strip() != '')]
        df.reset_index(drop=True, inplace=True)

        # Check if data was entirely removed during preprocessing
        if df.empty:
            self.logger.error("No valid rows after preprocessing. Exiting.")
            raise ValueError("All rows were removed during preprocessing")

        if self.deduplicate:
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['input_text', 'target_text'], keep='first').reset_index(drop=True)
            self.logger.info(f"Dropped {initial_rows - len(df)} duplicate rows")

            # Check if deduplication removed all rows
            if df.empty:
                self.logger.error("No valid rows after deduplication. Exiting.")
                raise ValueError("All rows removed during deduplication")

        # Lets augument that data so that our model does not overfit
        if self.augument_data:
            df = self.data_augumentation(df)

        return df  

    def data_augumentation(self, df:pd.DataFrame) -> pd.DataFrame:
        """Augment the dataset using various techniques."""
        augmented_data = []

        for _, row in df.iterrows():
            input_text = row['input_text']
            target_text = row['target_text']

            augmented_input = self.synonym_replacement(input_text)
            augmented_data.append({'input_text': augmented_input, 'target_text': target_text})

            augmented_input = self.random_insertion(input_text)
            augmented_data.append({'input_text': augmented_input, 'target_text': target_text})

            augmented_input = self.random_deletion(input_text)
            augmented_data.append({'input_text': augmented_input, 'target_text': target_text})

        augmented_df = pd.DataFrame(augmented_data)
        return pd.concat([df, augmented_df], ignore_index=True)

    def synonym_replacement(self, text:str) -> str:
        """Replace words with their synonyms."""
        words = text.split()
        new_words = words.copy()
        for i, word in enumerate(words):
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                new_words[i] = synonym
        return ' '.join(new_words)
    

    def random_insertion(self, text:str) ->str:
        """Insert random words into the text."""
        words = text.split()
        if len(words) < 2:
            return text
        new_word = random.choice(words)
        pos = random.randint(0, len(words))
        words.insert(pos, new_word)
        return ' '.join(words)

    def random_deletion(self, text:str, p:float=0.1) ->str:
        """Randomly delete words from the text."""
        words = text.split()
        if len(words) == 1:
            return text
        new_words = [word for word in words if random.random() > p]
        return ' '.join(new_words) if new_words else text
    
    def load_model_from_checkpoint(self, checkpoint_dir:str):
        """Load a pre-trained model from a specific checkpoint directory for inference"""
        self.logger.info(f"Loading model from checkpoint: {checkpoint_dir}")
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.logger.info("Model loaded successfully from checkpoint")

    def preprocess_text(self, text:str) -> str:
        """Normalize text by stripping, lowercasing, and emoji removal."""
        if self.remove_emojis:
            text = emoji.replace_emoji(text, '')
        text = re.sub(r'\s+', ' ', text.strip())
        return text[0].upper() + text[1:].lower() if text else text

    def split_data(self, df, train_size=0.8) -> Union[pd.DataFrame,pd.DataFrame]:        
        """Split data into training and testing sets."""
        self.logger.info("Splitting data")        
        train_df = df.sample(frac=train_size, random_state=42)
        test_df = df.drop(train_df.index).reset_index(drop=True)
        return train_df, test_df

    def tokenize_data(self, text_list: List[str]) -> Dict[str, torch.Tensor] :
        """Tokenize text batches into model-compatible tensors."""
        return self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
        )

    def prepare_datasets(self, train_df, test_df) -> Union[Dataset,Dataset]:
        """Convert DataFrames into PyTorch Datasets."""
        class DialectDataset(Dataset):
            def __init__(self, input_encodings, target_encodings):
                self.input_encodings = input_encodings
                self.target_encodings = target_encodings

            def __len__(self):
                return len(self.input_encodings["input_ids"])

            def __getitem__(self, idx):
                return {
                    "input_ids": self.input_encodings["input_ids"][idx],
                    "attention_mask": self.input_encodings["attention_mask"][idx],
                    "labels": self.target_encodings["input_ids"][idx],
                }

        train_inputs = self.tokenize_data(train_df["input_text"].tolist())
        train_targets = self.tokenize_data(train_df["target_text"].tolist())
        test_inputs = self.tokenize_data(test_df["input_text"].tolist())
        test_targets = self.tokenize_data(test_df["target_text"].tolist())

        train_dataset = DialectDataset(train_inputs, train_targets)
        test_dataset = DialectDataset(test_inputs, test_targets)
        return train_dataset, test_dataset

    def train(self, train_dataset, test_dataset, training_args:Seq2SeqTrainingArguments):
        """Train the model using Hugging Face Trainer."""
        self.logger.info("Training model")
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()
        return trainer

    def run(self, file_path:str, training_args:Seq2SeqTrainingArguments, train_size=0.8) -> Trainer:
        """Execute the entire pipeline from data loading to training."""
        df = self.load_data(file_path)
        df = self.preprocess_dataframe(df)
        
        train_df, test_df = self.split_data(df, train_size=train_size)

        train_dataset, test_dataset = self.prepare_datasets(train_df, test_df)

        trainer = self.train(train_dataset, test_dataset, training_args)
        return trainer

    def predict(self, input_texts) -> List[str]:
        """Generate predictions from preprocessed input texts."""
        input_ids = self.tokenize_data(input_texts).input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids, max_length=50)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

