import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

class ToxicSpansAnalyzer:
    def __init__(self, model_name: str, dataset_name: str = 'heegyu/toxic-spans'):
        """
        Initialize the ToxicSpansAnalyzer with a specific model and dataset.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name)
        
        # Initialize tokenizer with optimized settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            model_max_length=256
        )
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict:
        """
        Compute evaluation metrics for the model.
        
        Args:
            eval_pred: Evaluation prediction object containing predictions and labels
            
        Returns:
            Dictionary containing various metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def preprocess_dataset(self) -> Dict:
        """
        Preprocess the dataset for training.
        """
        def preprocess_function(examples):
            return self.tokenizer(
                examples["text_of_post"],
                truncation=True,
                padding="max_length",
                max_length=256,
                return_attention_mask=True,
                return_tensors=None
            )
        
        def preprocess_labels(examples):
            examples["labels"] = examples["toxic"]
            return examples
        
        logger.info("Starting dataset preprocessing...")
        
        tokenized_datasets = self.dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            num_proc=4
        )
        
        tokenized_datasets = tokenized_datasets.map(
            preprocess_labels,
            batched=True,
            num_proc=4
        )
        
        columns_to_remove = [
            "text_of_post", "toxic", "probability", "position",
            "type", "support", "position_probability"
        ]
        tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
        
        tokenized_datasets.set_format("torch")
        
        logger.info("Dataset preprocessing completed")
        return tokenized_datasets

    def save_model(self, output_dir: str):
        """
        Save the model and tokenizer properly.
        
        Args:
            output_dir: Directory to save the model
        """
        logger.info(f"Saving model to {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model with all components
        self.model.save_pretrained(
            output_dir,
            save_config=True,
            safe_serialization=True  # Use safe serialization for better compatibility
        )
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Model and tokenizer saved successfully")

    def push_to_hub(self, repo_name: str):
        """
        Push the model and tokenizer to the Hugging Face Hub.
        
        Args:
            repo_name: Name of the repository on the Hub
        """
        logger.info(f"Pushing model to Hub as {repo_name}")
        
        try:
            # Push both model and tokenizer
            self.model.push_to_hub(repo_name)
            self.tokenizer.push_to_hub(repo_name)
            logger.info("Successfully pushed model and tokenizer to Hub")
        except Exception as e:
            logger.error(f"Error pushing to Hub: {str(e)}")
            raise

    def fine_tune(self, output_dir: str, repo_name: str = None):
        """
        Fine-tune the model and save results.
        
        Args:
            output_dir: Directory to save the model
            repo_name: Optional name for the Hub repository
        """
        tokenized_datasets = self.preprocess_dataset()
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            torchscript=True
        ).to(self.device)
        
        # Calculate optimal batch size
        optimal_batch_size = 32
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            optimal_batch_size = int(min(32 * (gpu_memory / 8), 128))
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=optimal_batch_size,
            per_device_eval_batch_size=optimal_batch_size * 2,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            # Disable automatic Hub pushing (we'll do it manually)
            push_to_hub=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        
        # Log evaluation results
        logger.info("Evaluation Results:")
        for key, value in eval_results.items():
            logger.info(f"{key}: {value}")
        
        # Save the model properly
        self.save_model(output_dir)
        
        # Push to hub if repo_name is provided
        if repo_name:
            self.push_to_hub(repo_name)
        
        return train_result, eval_results

def run_experiment(models: List[str], test_texts: List[str], base_output_dir: str = "./results"):
    """
    Run experiments across multiple models.
    """
    results = {}
    
    for model_name in models:
        logger.info(f"Processing model: {model_name}")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            analyzer = ToxicSpansAnalyzer(model_name)
            output_dir = os.path.join(base_output_dir, model_name.replace('/', '_'))
            repo_name = f"toxic-spans-{model_name.split('/')[-1]}"
            
            train_results, eval_results = analyzer.fine_tune(output_dir, repo_name)
            
            results[model_name] = {
                "train_results": train_results,
                "eval_results": eval_results
            }
            
        except Exception as e:
            logger.error(f"Error processing model {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
        
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    # Example usage
    test_texts = [
        "This is a toxic piece of text with some harmful content.",
        "This is a normal, non-toxic piece of text.",
        "Another example with potentially toxic elements."
    ]
    
    # Set torch multiprocessing method
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Define model lists
    BERT_MODELS = [
        "lyeonii/bert-tiny",
        "lyeonii/bert-small",
        "lyeonii/bert-medium",
        "google-bert/bert-base-uncased",
        "google-bert/bert-large-uncased"
    ]
    
    ROBERTA_MODELS = [
        "smallbenchnlp/roberta-small",
        "JackBAI/roberta-medium",
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large"
    ]
    
    # Run experiments
    logger.info("Starting BERT experiments...")
    bert_results = run_experiment(BERT_MODELS, test_texts)
    
    logger.info("Starting RoBERTa experiments...")
    roberta_results = run_experiment(ROBERTA_MODELS, test_texts)
    
    # Print final results
    logger.info("Experiment completed")