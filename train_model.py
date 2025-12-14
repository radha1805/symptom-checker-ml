"""
Model training script for symptoms checker project.
Trains ML models and provides prediction functionality.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    hamming_loss, classification_report
)
from tabulate import tabulate

# Import our custom modules
from feature_engineer import build_training_matrices
from symptom_extractor import extract_symptoms


class SymptomPredictor:
    """
    Handles model training, evaluation, and prediction for symptom-based disease diagnosis.
    """
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        """
        Initialize the symptom predictor.
        
        Args:
            artifacts_dir: Directory containing artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = self.artifacts_dir / "model.joblib"
        self.meta_path = self.artifacts_dir / "meta.json"
        
        # Load metadata
        self.meta_data = self._load_meta_data()
        self.vocab = self.meta_data.get("vocab", [])
        self.diseases = self.meta_data.get("diseases", [])
        self.threshold = self.meta_data.get("threshold", 0.3)
        
        # Model and feature engineer
        self.model = None
        self.feature_engineer = None
        
        print(f"✓ Loaded metadata: {len(self.vocab)} symptoms, {len(self.diseases)} diseases")
    
    def _load_meta_data(self) -> Dict[str, Any]:
        """
        Load metadata from JSON file.
        
        Returns:
            Dictionary containing metadata
        """
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta data not found: {self.meta_path}")
        
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_model(self) -> OneVsRestClassifier:
        """
        Create the ML model pipeline.
        
        Returns:
            Configured OneVsRestClassifier with RandomForestClassifier
        """
        # Base classifier with balanced class weights and parallel processing
        base_classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        # Wrap in OneVsRest for multi-label classification
        model = OneVsRestClassifier(base_classifier)
        
        print("✓ Created OneVsRestClassifier with RandomForestClassifier")
        return model
    
    def train_model(self) -> Dict[str, Any]:
        """
        Train the ML model and evaluate performance.
        
        Returns:
            Dictionary containing training results and metrics
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Load training data
        print("Loading training matrices...")
        X, Y, vocab_list, diseases_list = build_training_matrices(str(self.artifacts_dir))
        
        # Store feature engineer for later use
        from feature_engineer import create_feature_engineer
        self.feature_engineer = create_feature_engineer(str(self.artifacts_dir))
        
        print(f"Training data: X shape {X.shape}, Y shape {Y.shape}")
        
        # Split data into train/test sets (80/20 split with fixed seed)
        print("Splitting data into train/test sets (80/20)...")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"Train set: X_train {X_train.shape}, Y_train {Y_train.shape}")
        print(f"Test set: X_test {X_test.shape}, Y_test {Y_test.shape}")
        
        # Create and train model
        print("\nTraining model...")
        self.model = self._create_model()
        
        # Train the model
        self.model.fit(X_train, Y_train)
        print("✓ Model training completed")
        
        # Make predictions on test set
        print("Making predictions on test set...")
        Y_pred = self.model.predict(X_test)
        Y_pred_proba = self.model.predict_proba(X_test)
        
        # Compute evaluation metrics
        print("Computing evaluation metrics...")
        metrics = self._compute_metrics(Y_test, Y_pred, Y_pred_proba)
        
        # Save trained model
        self._save_model()
        
        # Update metadata with training results
        self._update_meta_data(metrics)
        
        return metrics
    
    def _compute_metrics(self, Y_true: np.ndarray, Y_pred: np.ndarray, Y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            Y_true: True labels
            Y_pred: Predicted labels
            Y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary containing all computed metrics
        """
        # Overall metrics
        hamming_loss_score = hamming_loss(Y_true, Y_pred)
        macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
        
        # Per-disease metrics
        precision_per_disease = precision_score(Y_true, Y_pred, average=None, zero_division=0)
        recall_per_disease = recall_score(Y_true, Y_pred, average=None, zero_division=0)
        f1_per_disease = f1_score(Y_true, Y_pred, average=None, zero_division=0)
        
        # Create metrics summary
        metrics = {
            "overall": {
                "hamming_loss": float(hamming_loss_score),
                "macro_f1": float(macro_f1),
                "micro_f1": float(micro_f1)
            },
            "per_disease": {
                "precision": precision_per_disease.tolist(),
                "recall": recall_per_disease.tolist(),
                "f1_score": f1_per_disease.tolist()
            }
        }
        
        # Print metrics table
        self._print_metrics_table(metrics)
        
        return metrics
    
    def _print_metrics_table(self, metrics: Dict[str, Any]) -> None:
        """
        Print a neat table of evaluation metrics.
        
        Args:
            metrics: Dictionary containing computed metrics
        """
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        # Overall metrics
        overall = metrics["overall"]
        print(f"Hamming Loss: {overall['hamming_loss']:.4f}")
        print(f"Macro F1-Score: {overall['macro_f1']:.4f}")
        print(f"Micro F1-Score: {overall['micro_f1']:.4f}")
        
        # Per-disease metrics table
        print(f"\nPer-Disease Metrics:")
        print("-" * 80)
        
        # Prepare data for table
        table_data = []
        for i, disease in enumerate(self.diseases):
            precision = metrics["per_disease"]["precision"][i]
            recall = metrics["per_disease"]["recall"][i]
            f1 = metrics["per_disease"]["f1_score"][i]
            
            table_data.append([
                disease[:30] + "..." if len(disease) > 30 else disease,
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{f1:.3f}"
            ])
        
        # Print table
        headers = ["Disease", "Precision", "Recall", "F1-Score"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Summary statistics
        precision_scores = metrics["per_disease"]["precision"]
        recall_scores = metrics["per_disease"]["recall"]
        f1_scores = metrics["per_disease"]["f1_score"]
        
        print(f"\nSummary Statistics:")
        print(f"Mean Precision: {np.mean(precision_scores):.3f}")
        print(f"Mean Recall: {np.mean(recall_scores):.3f}")
        print(f"Mean F1-Score: {np.mean(f1_scores):.3f}")
        print("="*80)
    
    def _save_model(self) -> None:
        """
        Save the trained model and metadata to artifacts directory.
        """
        print("Saving trained model...")
        
        # Prepare model data for saving
        model_data = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "vocab": self.vocab,
            "diseases": self.diseases,
            "threshold": self.threshold,
            "meta_data": self.meta_data
        }
        
        # Save to joblib file
        joblib.dump(model_data, self.model_path)
        print(f"✓ Model saved to: {self.model_path}")
    
    def _update_meta_data(self, metrics: Dict[str, Any]) -> None:
        """
        Update metadata with training results.
        
        Args:
            metrics: Dictionary containing training metrics
        """
        print("Updating metadata...")
        
        # Update metadata with training results
        self.meta_data.update({
            "vocab": self.vocab,
            "diseases": self.diseases,
            "threshold": self.threshold,
            "version": "v1",
            "best_params": "RandomForestClassifier(n_estimators=200, class_weight='balanced')",
            "training_metrics": {
                "hamming_loss": metrics["overall"]["hamming_loss"],
                "macro_f1": metrics["overall"]["macro_f1"],
                "micro_f1": metrics["overall"]["micro_f1"]
            }
        })
        
        # Save updated metadata
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Metadata updated: {self.meta_path}")
    
    def predict_from_text(self, text: str, lang: str = "en") -> List[Tuple[str, float]]:
        """
        Predict diseases from text input.
        
        Args:
            text: Input text containing symptoms
            lang: Language code (currently only English supported)
            
        Returns:
            List of tuples (disease_name, probability) sorted by probability (descending)
        """
        if not self.model:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Extract symptoms from text
        symptoms = extract_symptoms(text)
        
        if not symptoms:
            return []
        
        # Build feature vector
        feature_vector = self.feature_engineer.build_feature_vector(symptoms)
        
        # Reshape for prediction (sklearn expects 2D array)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Create disease-probability pairs
        disease_probs = [(disease, prob) for disease, prob in zip(self.diseases, probabilities)]
        
        # Sort by probability (descending) and filter by threshold
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top predictions above threshold
        top_predictions = [(disease, prob) for disease, prob in disease_probs if prob >= self.threshold]
        
        return top_predictions[:3]  # Return top 3 predictions
    
    def load_model(self) -> None:
        """
        Load a previously trained model from artifacts directory.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print("Loading trained model...")
        model_data = joblib.load(self.model_path)
        
        self.model = model_data["model"]
        self.feature_engineer = model_data["feature_engineer"]
        self.vocab = model_data["vocab"]
        self.diseases = model_data["diseases"]
        self.threshold = model_data["threshold"]
        
        print("✓ Model loaded successfully")


def train_and_evaluate_model(artifacts_dir: str = "artifacts") -> SymptomPredictor:
    """
    Train and evaluate the symptom prediction model.
    
    Args:
        artifacts_dir: Directory containing artifacts
        
    Returns:
        Trained SymptomPredictor instance
    """
    predictor = SymptomPredictor(artifacts_dir)
    predictor.train_model()
    return predictor


def predict_from_text(text: str, lang: str = "en", artifacts_dir: str = "artifacts") -> List[Tuple[str, float]]:
    """
    Convenience function to predict diseases from text.
    
    Args:
        text: Input text containing symptoms
        lang: Language code
        artifacts_dir: Directory containing artifacts
        
    Returns:
        List of tuples (disease_name, probability)
    """
    predictor = SymptomPredictor(artifacts_dir)
    predictor.load_model()
    return predictor.predict_from_text(text, lang)


if __name__ == "__main__":
    """
    Main execution: train model and test prediction.
    """
    print("Symptom Checker - Model Training")
    print("=" * 50)
    
    try:
        # Train the model
        predictor = train_and_evaluate_model()
        
        # Test prediction with sample text
        print("\n" + "="*60)
        print("TESTING PREDICTION")
        print("="*60)
        
        test_text = "i have fever and sore throat and chills"
        print(f"Input text: \"{test_text}\"")
        
        # Extract symptoms first
        symptoms = extract_symptoms(test_text)
        print(f"Extracted symptoms: {symptoms}")
        
        # Make prediction
        predictions = predictor.predict_from_text(test_text)
        
        print(f"\nTop-5 Disease Predictions:")
        print("-" * 40)
        
        if predictions:
            for i, (disease, probability) in enumerate(predictions, 1):
                print(f"{i}. {disease}: {probability:.3f}")
        else:
            print("No diseases predicted above threshold.")
        
        # Also show top 5 predictions regardless of threshold
        print(f"\nTop-5 Disease Predictions (all probabilities):")
        print("-" * 50)
        
        # Get all predictions sorted by probability
        symptoms = extract_symptoms(test_text)
        feature_vector = predictor.feature_engineer.build_feature_vector(symptoms)
        feature_vector = feature_vector.reshape(1, -1)
        probabilities = predictor.model.predict_proba(feature_vector)[0]
        
        all_predictions = [(disease, prob) for disease, prob in zip(predictor.diseases, probabilities)]
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        for i, (disease, probability) in enumerate(all_predictions[:5], 1):
            print(f"{i}. {disease}: {probability:.3f}")
        
        print("\n✓ Model training and testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training/testing: {str(e)}")
        raise
