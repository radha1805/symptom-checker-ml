"""
Model evaluation script for symptoms checker project.
Performs cross-validation, hyperparameter tuning, and confusion analysis.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    hamming_loss, classification_report
)
from tabulate import tabulate
from scipy.stats import randint

# Import our custom modules
from feature_engineer import build_training_matrices


class ModelEvaluator:
    """
    Handles comprehensive model evaluation including cross-validation,
    hyperparameter tuning, and confusion analysis.
    """
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        """
        Initialize the model evaluator.
        
        Args:
            artifacts_dir: Directory containing artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = self.artifacts_dir / "model.joblib"
        self.meta_path = self.artifacts_dir / "meta.json"
        
        # Load model and metadata
        self.model_data = self._load_model()
        self.meta_data = self._load_meta_data()
        
        # Extract components
        self.model = self.model_data["model"]
        self.vocab = self.meta_data.get("vocab", [])
        self.diseases = self.meta_data.get("diseases", [])
        self.threshold = self.meta_data.get("threshold", 0.3)
        
        print(f"✓ Loaded model with {len(self.vocab)} symptoms and {len(self.diseases)} diseases")
    
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the trained model from joblib file.
        
        Returns:
            Dictionary containing model and metadata
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        return joblib.load(self.model_path)
    
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
    
    def _save_meta_data(self) -> None:
        """
        Save updated metadata to JSON file.
        """
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Updated metadata: {self.meta_path}")
    
    def run_cross_validation(self, cv_folds: int = 5) -> Dict[str, float]:
        """
        Run k-fold cross-validation on the model.
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing average metrics across folds
        """
        print(f"\n{'='*60}")
        print(f"RUNNING {cv_folds}-FOLD CROSS-VALIDATION")
        print(f"{'='*60}")
        
        # Load training data
        print("Loading training matrices...")
        X, Y, _, _ = build_training_matrices(str(self.artifacts_dir))
        
        print(f"Cross-validation data: X shape {X.shape}, Y shape {Y.shape}")
        
        # Create stratified k-fold for balanced splits
        # Note: For multi-label data, we'll use regular KFold since StratifiedKFold
        # doesn't work well with multi-label targets
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Store metrics for each fold
        fold_metrics = {
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'precision_micro': [],
            'recall_micro': [],
            'f1_micro': [],
            'hamming_loss': []
        }
        
        print(f"\nRunning {cv_folds}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"  Fold {fold}/{cv_folds}...")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            Y_train_fold, Y_val_fold = Y[train_idx], Y[val_idx]
            
            # Create and train model for this fold
            fold_model = self._create_model()
            fold_model.fit(X_train_fold, Y_train_fold)
            
            # Make predictions
            Y_pred_fold = fold_model.predict(X_val_fold)
            
            # Compute metrics
            precision_macro = precision_score(Y_val_fold, Y_pred_fold, average='macro', zero_division=0)
            recall_macro = recall_score(Y_val_fold, Y_pred_fold, average='macro', zero_division=0)
            f1_macro = f1_score(Y_val_fold, Y_pred_fold, average='macro', zero_division=0)
            
            precision_micro = precision_score(Y_val_fold, Y_pred_fold, average='micro', zero_division=0)
            recall_micro = recall_score(Y_val_fold, Y_pred_fold, average='micro', zero_division=0)
            f1_micro = f1_score(Y_val_fold, Y_pred_fold, average='micro', zero_division=0)
            
            hamming_loss_score = hamming_loss(Y_val_fold, Y_pred_fold)
            
            # Store metrics
            fold_metrics['precision_macro'].append(precision_macro)
            fold_metrics['recall_macro'].append(recall_macro)
            fold_metrics['f1_macro'].append(f1_macro)
            fold_metrics['precision_micro'].append(precision_micro)
            fold_metrics['recall_micro'].append(recall_micro)
            fold_metrics['f1_micro'].append(f1_micro)
            fold_metrics['hamming_loss'].append(hamming_loss_score)
        
        # Compute average metrics
        avg_metrics = {}
        for metric_name, values in fold_metrics.items():
            avg_metrics[metric_name] = np.mean(values)
            std_metrics = np.std(values)
            print(f"  {metric_name}: {avg_metrics[metric_name]:.4f} ± {std_metrics:.4f}")
        
        # Print summary table
        self._print_cv_summary_table(avg_metrics)
        
        return avg_metrics
    
    def _create_model(self) -> OneVsRestClassifier:
        """
        Create a model instance for cross-validation.
        
        Returns:
            Configured OneVsRestClassifier
        """
        base_classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        return OneVsRestClassifier(base_classifier)
    
    def _print_cv_summary_table(self, metrics: Dict[str, float]) -> None:
        """
        Print a summary table of cross-validation results.
        
        Args:
            metrics: Dictionary containing average metrics
        """
        print(f"\nCross-Validation Summary:")
        print("-" * 50)
        
        table_data = [
            ["Macro Precision", f"{metrics['precision_macro']:.4f}"],
            ["Macro Recall", f"{metrics['recall_macro']:.4f}"],
            ["Macro F1-Score", f"{metrics['f1_macro']:.4f}"],
            ["Micro Precision", f"{metrics['precision_micro']:.4f}"],
            ["Micro Recall", f"{metrics['recall_micro']:.4f}"],
            ["Micro F1-Score", f"{metrics['f1_micro']:.4f}"],
            ["Hamming Loss", f"{metrics['hamming_loss']:.4f}"]
        ]
        
        print(tabulate(table_data, headers=["Metric", "Average Score"], tablefmt="grid"))
    
    def run_hyperparameter_tuning(self, n_iter: int = 10) -> Dict[str, Any]:
        """
        Run randomized grid search for hyperparameter tuning.
        
        Args:
            n_iter: Number of random parameter combinations to try
            
        Returns:
            Dictionary containing best parameters and scores
        """
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING (Randomized Search)")
        print(f"{'='*60}")
        
        # Load training data
        print("Loading training matrices...")
        X, Y, _, _ = build_training_matrices(str(self.artifacts_dir))
        
        print(f"Tuning data: X shape {X.shape}, Y shape {Y.shape}")
        
        # Define parameter grid
        param_distributions = {
            'estimator__n_estimators': [100, 200, 400],
            'estimator__max_depth': [None, 10, 20]
        }
        
        print(f"Parameter grid: {param_distributions}")
        print(f"Number of iterations: {n_iter}")
        
        # Create base model
        base_classifier = RandomForestClassifier(
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        model = OneVsRestClassifier(base_classifier)
        
        # Create randomized search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=3,  # Use 3-fold CV for speed
            scoring='f1_macro',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"\nRunning randomized search...")
        random_search.fit(X, Y)
        
        # Extract results
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best cross-validation score: {best_score:.4f}")
        
        # Update metadata with best parameters
        self.meta_data['best_params'] = best_params
        self.meta_data['best_cv_score'] = float(best_score)
        self._save_meta_data()
        
        # Print parameter importance analysis
        self._print_parameter_analysis(random_search.cv_results_)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': random_search.cv_results_
        }
    
    def _print_parameter_analysis(self, cv_results: Dict[str, Any]) -> None:
        """
        Print analysis of parameter combinations tried.
        
        Args:
            cv_results: Results from RandomizedSearchCV
        """
        print(f"\nParameter Analysis:")
        print("-" * 40)
        
        # Extract parameter values and scores
        param_names = [key for key in cv_results.keys() if key.startswith('param_')]
        scores = cv_results['mean_test_score']
        
        # Create analysis table
        analysis_data = []
        for i in range(len(scores)):
            row = [f"{scores[i]:.4f}"]
            for param_name in param_names:
                param_value = cv_results[param_name][i]
                row.append(str(param_value))
            analysis_data.append(row)
        
        # Sort by score (descending)
        analysis_data.sort(key=lambda x: float(x[0]), reverse=True)
        
        headers = ["Score"] + [name.replace('param_', '').replace('estimator__', '') for name in param_names]
        print(tabulate(analysis_data[:5], headers=headers, tablefmt="grid"))
    
    def analyze_confusion_patterns(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Analyze confusion patterns - which diseases are most confused with each other.
        
        Returns:
            Dictionary mapping each disease to its top 3 most confused-with diseases
        """
        print(f"\n{'='*60}")
        print(f"CONFUSION PATTERN ANALYSIS")
        print(f"{'='*60}")
        
        # Load training data
        print("Loading training matrices...")
        X, Y, _, _ = build_training_matrices(str(self.artifacts_dir))
        
        print(f"Analysis data: X shape {X.shape}, Y shape {Y.shape}")
        
        # Get prediction probabilities for all diseases
        print("Computing prediction probabilities...")
        Y_proba = self.model.predict_proba(X)
        
        confusion_patterns = {}
        
        print(f"\nAnalyzing confusion patterns for each disease...")
        
        for disease_idx, disease in enumerate(self.diseases):
            # Get true labels for this disease
            true_labels = Y[:, disease_idx]
            
            # Get prediction probabilities for this disease
            pred_proba = Y_proba[:, disease_idx]
            
            # Find cases where this disease was NOT the true label
            wrong_cases = np.where(true_labels == 0)[0]
            
            if len(wrong_cases) == 0:
                confusion_patterns[disease] = []
                continue
            
            # Get probabilities for wrong cases
            wrong_proba = pred_proba[wrong_cases]
            
            # Find which diseases were predicted with highest probability
            # for cases where this disease was NOT the true label
            confusion_scores = {}
            
            for wrong_case_idx in wrong_cases:
                # Get probabilities for this wrong case
                case_proba = Y_proba[wrong_case_idx]
                
                # Find the disease with highest probability (excluding the true disease)
                # Get true disease for this case
                true_disease_idx = np.argmax(Y[wrong_case_idx])
                
                # Get probabilities excluding the true disease
                other_proba = np.copy(case_proba)
                other_proba[true_disease_idx] = 0  # Exclude true disease
                
                # Find disease with highest probability among wrong predictions
                predicted_disease_idx = np.argmax(other_proba)
                predicted_disease = self.diseases[predicted_disease_idx]
                predicted_prob = other_proba[predicted_disease_idx]
                
                # Accumulate confusion scores
                if predicted_disease not in confusion_scores:
                    confusion_scores[predicted_disease] = []
                confusion_scores[predicted_disease].append(predicted_prob)
            
            # Calculate average confusion scores
            avg_confusion_scores = {}
            for confused_disease, scores in confusion_scores.items():
                avg_confusion_scores[confused_disease] = np.mean(scores)
            
            # Sort by average confusion score and get top 3
            sorted_confusions = sorted(
                avg_confusion_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            confusion_patterns[disease] = sorted_confusions
        
        # Print confusion analysis
        self._print_confusion_analysis(confusion_patterns)
        
        return confusion_patterns
    
    def _print_confusion_analysis(self, confusion_patterns: Dict[str, List[Tuple[str, float]]]) -> None:
        """
        Print confusion pattern analysis in a readable format.
        
        Args:
            confusion_patterns: Dictionary containing confusion patterns
        """
        print(f"\nConfusion Pattern Analysis:")
        print("=" * 80)
        print("For each disease, shows the top 3 diseases it's most confused with")
        print("(i.e., diseases that get highest prediction probabilities when")
        print("the model incorrectly predicts them instead of the true disease)")
        print("=" * 80)
        
        # Create table data
        table_data = []
        for disease, confusions in confusion_patterns.items():
            if confusions:
                confusion_str = ", ".join([f"{conf_disease} ({prob:.3f})" 
                                         for conf_disease, prob in confusions])
            else:
                confusion_str = "No confusion data"
            
            table_data.append([
                disease[:25] + "..." if len(disease) > 25 else disease,
                confusion_str
            ])
        
        # Print table
        headers = ["Disease", "Top 3 Most Confused With (avg probability)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print interpretation guide
        print(f"\nInterpretation Guide:")
        print("-" * 50)
        print("• Higher probabilities indicate stronger confusion patterns")
        print("• Diseases with similar symptoms tend to be confused more often")
        print("• This analysis helps identify which diseases need better")
        print("  feature engineering or additional training data")
        print("• Consider collecting more distinguishing symptoms for")
        print("  diseases with high confusion scores")


def run_full_evaluation(artifacts_dir: str = "artifacts") -> Dict[str, Any]:
    """
    Run complete model evaluation including CV, hyperparameter tuning, and confusion analysis.
    
    Args:
        artifacts_dir: Directory containing artifacts
        
    Returns:
        Dictionary containing all evaluation results
    """
    print("Symptom Checker - Model Evaluation")
    print("=" * 50)
    
    evaluator = ModelEvaluator(artifacts_dir)
    
    # Run cross-validation
    cv_results = evaluator.run_cross_validation(cv_folds=5)
    
    # Run hyperparameter tuning
    tuning_results = evaluator.run_hyperparameter_tuning(n_iter=10)
    
    # Analyze confusion patterns
    confusion_results = evaluator.analyze_confusion_patterns()
    
    # Compile all results
    all_results = {
        'cross_validation': cv_results,
        'hyperparameter_tuning': tuning_results,
        'confusion_analysis': confusion_results
    }
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print("✓ Cross-validation completed")
    print("✓ Hyperparameter tuning completed")
    print("✓ Confusion analysis completed")
    print("✓ Best parameters saved to meta.json")
    
    return all_results


if __name__ == "__main__":
    """
    Main execution: run complete model evaluation.
    """
    try:
        results = run_full_evaluation()
        
        print(f"\nSummary of Results:")
        print("-" * 30)
        print(f"Best CV F1-Score: {results['hyperparameter_tuning']['best_score']:.4f}")
        print(f"Best Parameters: {results['hyperparameter_tuning']['best_params']}")
        print(f"Average CV Hamming Loss: {results['cross_validation']['hamming_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise
