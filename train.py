import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
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

class ToxicSpansAnalyzer:
    def __init__(self, model_name: str, dataset_name: str = 'heegyu/toxic-spans'):
        """
        Initialize the ToxicSpansAnalyzer with a specific model and dataset.
        
        Args:
            model_name: Name of the pretrained model to use
            dataset_name: Name of the dataset to use for fine-tuning
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def preprocess_dataset(self) -> Dict:
        """
        Preprocess the dataset for training.
        
        Returns:
            Preprocessed dataset dictionary
        """
        def preprocess_function(examples):
            return self.tokenizer(
                examples["text_of_post"],
                truncation=True,
                padding="max_length",
                max_length=128,
                return_attention_mask=True
            )
        
        def preprocess_labels(examples):
            examples["labels"] = examples["toxic"]
            return examples
        
        # Process the dataset
        tokenized_datasets = self.dataset.map(preprocess_function, batched=True)
        tokenized_datasets = tokenized_datasets.map(preprocess_labels)
        
        # Remove unnecessary columns
        columns_to_remove = [
            "text_of_post", "toxic", "probability", "position",
            "type", "support", "position_probability"
        ]
        tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
        
        return tokenized_datasets

    def fine_tune(self, output_dir: str):
        """
        Fine-tune the model on the toxic spans dataset.
        
        Args:
            output_dir: Directory to save the model and results
        """
        tokenized_datasets = self.preprocess_dataset()
        
        # Initialize model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,  # Binary classification
        ).to(self.device)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            push_to_hub=True,
            hub_model_id=f"toxic-spans-{self.model_name.split('/')[-1]}"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=self.tokenizer,
        )
        
        # Train and push to hub
        trainer.train()
        trainer.push_to_hub()
        
        return trainer

    def calculate_token_importance(self, text: str) -> Tuple[List[float], List[str]]:
        """
        Calculate importance scores for each token in the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (importance_scores, tokens)
        """
        # Tokenize input
        tokens = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Get model outputs with gradient computation
        self.model.eval()
        inputs['input_ids'].requires_grad_()
        outputs = self.model(**inputs)
        
        # Calculate gradients with respect to toxic class
        toxic_class_score = outputs.logits[0, 1]  # Get score for toxic class
        toxic_class_score.backward()
        
        # Get gradients and token importances
        input_grads = inputs['input_ids'].grad[0].abs()
        importance_scores = input_grads.cpu().numpy()
        
        return importance_scores, tokens

    def compute_metrics(self, text: str) -> Dict[str, float]:
        """
        Compute metrics for token importance analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of computed metrics
        """
        importance_scores, tokens = self.calculate_token_importance(text)
        
        # Calculate various metrics
        metrics = {
            'mean_importance': float(np.mean(importance_scores)),
            'max_importance': float(np.max(importance_scores)),
            'importance_std': float(np.std(importance_scores)),
            'top_10_percent_ratio': float(np.mean(importance_scores >= np.percentile(importance_scores, 90))),
        }
        
        return metrics

def run_experiment(models: List[str], test_texts: List[str], base_output_dir: str = "./results"):
    """
    Run the full experiment across multiple models.
    
    Args:
        models: List of model names to evaluate
        test_texts: List of texts to use for evaluation
        base_output_dir: Base directory for saving results
    """
    results = {}
    
    for model_name in models:
        logger.info(f"Processing model: {model_name}")
        
        try:
            # Initialize analyzer
            analyzer = ToxicSpansAnalyzer(model_name)
            
            # Fine-tune model
            output_dir = os.path.join(base_output_dir, model_name.replace('/', '_'))
            analyzer.fine_tune(output_dir)
            
            # Evaluate on test texts
            model_results = {}
            for text in test_texts:
                metrics = analyzer.compute_metrics(text)
                for metric_name, value in metrics.items():
                    if metric_name not in model_results:
                        model_results[metric_name] = []
                    model_results[metric_name].append(value)
            
            # Average metrics across texts
            results[model_name] = {
                metric: float(np.mean(values))
                for metric, values in model_results.items()
            }
            
        except Exception as e:
            logger.error(f"Error processing model {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results

if __name__ == "__main__":
    # Example usage
    test_texts = [
        "This is a toxic piece of text with some harmful content.",
        "This is a normal, non-toxic piece of text.",
        "Another example with potentially toxic elements."
    ]
    
    # Run experiments for both BERT and RoBERTa models
    bert_results = run_experiment(BERT_MODELS, test_texts)
    roberta_results = run_experiment(ROBERTA_MODELS, test_texts)
    
    # Print results