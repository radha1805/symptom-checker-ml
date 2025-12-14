"""
Symptom extraction module for symptoms checker project.
Extracts symptoms from text using exact and fuzzy matching techniques.
"""

import json
import re
import string
from pathlib import Path
from typing import List, Tuple, Set
from rapidfuzz import fuzz, process


class SymptomExtractor:
    """
    Extracts symptoms from text using vocabulary-based matching.
    Supports both exact matching and fuzzy matching for typo tolerance.
    """
    
    def __init__(self, vocab_path: str = "artifacts/symptom_vocab.json"):
        """
        Initialize the symptom extractor with vocabulary.
        
        Args:
            vocab_path: Path to the symptom vocabulary JSON file
        """
        self.vocab_path = Path(vocab_path)
        self.symptom_vocab = self._load_vocabulary()
        self.fuzzy_threshold = 85  # Minimum similarity ratio for fuzzy matching
        
    def _load_vocabulary(self) -> List[str]:
        """
        Load symptom vocabulary from JSON file.
        
        Returns:
            List of canonical symptom strings
            
        Raises:
            FileNotFoundError: If vocabulary file doesn't exist
            json.JSONDecodeError: If JSON file is malformed
        """
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Symptom vocabulary not found: {self.vocab_path}")
        
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            if not isinstance(vocab, list):
                raise ValueError("Vocabulary must be a list of strings")
            
            print(f"✓ Loaded {len(vocab)} symptoms from vocabulary")
            return vocab
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in vocabulary file: {e}")
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for symptom matching.
        - Convert to lowercase
        - Remove punctuation except hyphens inside words
        - Replace multiple whitespace with single space
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove punctuation except hyphens that are inside words
        # Keep hyphens that are surrounded by word characters
        normalized = re.sub(r'(?<!\w)-(?!\w)', ' ', normalized)  # Remove standalone hyphens
        normalized = re.sub(r'[^\w\s-]', ' ', normalized)  # Remove other punctuation
        
        # Replace multiple whitespace with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _exact_match_symptoms(self, normalized_text: str) -> Set[str]:
        """
        Find exact matches for symptoms in the text using word boundaries.
        
        Args:
            normalized_text: Normalized input text
            
        Returns:
            Set of matched canonical symptom strings
        """
        matched_symptoms = set()
        
        for symptom in self.symptom_vocab:
            # Create regex pattern with word boundaries for multi-word symptoms
            # Escape special regex characters in the symptom
            escaped_symptom = re.escape(symptom)
            pattern = r'\b' + escaped_symptom + r'\b'
            
            if re.search(pattern, normalized_text):
                matched_symptoms.add(symptom)
        
        return matched_symptoms
    
    def _fuzzy_match_symptoms(self, normalized_text: str, exclude_exact: Set[str]) -> Set[str]:
        """
        Find fuzzy matches for symptoms not found by exact matching.
        
        Args:
            normalized_text: Normalized input text
            exclude_exact: Set of symptoms already found by exact matching
            
        Returns:
            Set of matched canonical symptom strings
        """
        matched_symptoms = set()
        
        # Split text into words for fuzzy matching
        words = normalized_text.split()
        
        # Only consider symptoms not already matched exactly
        remaining_symptoms = [s for s in self.symptom_vocab if s not in exclude_exact]
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
                
            # Find best fuzzy match for this word
            best_match = process.extractOne(
                word, 
                remaining_symptoms, 
                scorer=fuzz.ratio,
                score_cutoff=self.fuzzy_threshold
            )
            
            if best_match:
                matched_symptoms.add(best_match[0])
        
        return matched_symptoms
    
    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extract symptoms from input text using exact and fuzzy matching.
        
        Args:
            text: Input text to extract symptoms from
            
        Returns:
            List of canonical symptom strings found in the text
        """
        if not text or not text.strip():
            return []
        
        # Normalize the input text
        normalized_text = self._normalize_text(text)
        
        if not normalized_text:
            return []
        
        # First try exact matching
        exact_matches = self._exact_match_symptoms(normalized_text)
        
        # Then try fuzzy matching for remaining symptoms
        fuzzy_matches = self._fuzzy_match_symptoms(normalized_text, exact_matches)
        
        # Combine and deduplicate results
        all_matches = exact_matches.union(fuzzy_matches)
        
        # Return as sorted list for consistent output
        return sorted(list(all_matches))
    
    def extract_from_audio(self, audio_bytes: bytes, lang: str = "en") -> Tuple[str, List[str]]:
        """
        Extract symptoms from audio input.
        
        This function raises NotImplementedError to instruct the API layer
        to use the STT/TTS module for audio processing.
        
        Args:
            audio_bytes: Raw audio data
            lang: Language code for speech recognition
            
        Returns:
            Tuple of (transcribed_text, extracted_symptoms)
            
        Raises:
            NotImplementedError: Always raised to redirect to STT/TTS module
        """
        raise NotImplementedError(
            "Audio processing not implemented in symptom_extractor. "
            "Please use the STT/TTS module (stt_tts.py) to convert audio to text first, "
            "then call extract_symptoms() on the transcribed text."
        )


def create_extractor() -> SymptomExtractor:
    """
    Create and return a SymptomExtractor instance.
    
    Returns:
        Configured SymptomExtractor instance
    """
    return SymptomExtractor()


def extract_symptoms(text: str) -> List[str]:
    """
    Convenience function to extract symptoms from text.
    
    Args:
        text: Input text to extract symptoms from
        
    Returns:
        List of canonical symptom strings found in the text
    """
    extractor = create_extractor()
    return extractor.extract_symptoms(text)


def extract_from_audio(audio_bytes: bytes, lang: str = "en") -> Tuple[str, List[str]]:
    """
    Convenience function for audio symptom extraction.
    
    Args:
        audio_bytes: Raw audio data
        lang: Language code for speech recognition
        
    Returns:
        Tuple of (transcribed_text, extracted_symptoms)
        
    Raises:
        NotImplementedError: Always raised to redirect to STT/TTS module
    """
    extractor = create_extractor()
    return extractor.extract_from_audio(audio_bytes, lang)


if __name__ == "__main__":
    """
    Test examples demonstrating symptom extraction functionality.
    """
    print("Testing Symptom Extractor...")
    print("=" * 50)
    
    try:
        # Create extractor instance
        extractor = create_extractor()
        
        # Test cases with various scenarios
        test_cases = [
            "I have a fever and headache",
            "My stomach hurts and I feel nauseous",
            "I'm experiencing chest pain and shortness of breath",
            "I have a skin rash with itching",
            "I feel dizzy and have been vomiting",
            "I have joint pain and muscle weakness",
            "I'm sneezing continuously and have a runny nose",
            "I have back pain and difficulty sleeping",
            "I feel fatigued and have lost my appetite",
            "I have abdominal pain and constipation"
        ]
        
        print("Testing symptom extraction on sample sentences:")
        print("-" * 50)
        
        for i, test_text in enumerate(test_cases, 1):
            symptoms = extractor.extract_symptoms(test_text)
            print(f"{i:2d}. Text: \"{test_text}\"")
            print(f"    Symptoms: {symptoms}")
            print()
        
        # Test edge cases
        print("Testing edge cases:")
        print("-" * 50)
        
        edge_cases = [
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            ("I feel fine", "No symptoms"),
            ("I have fever, headache, and nausea", "Multiple symptoms with commas"),
            ("I have fever/headache", "Symptoms with slash separator"),
            ("I have a high fever", "Symptom with adjective"),
            ("I have been having headaches", "Symptom with verb form"),
        ]
        
        for test_text, description in edge_cases:
            symptoms = extractor.extract_symptoms(test_text)
            print(f"• {description}: \"{test_text}\"")
            print(f"  Symptoms: {symptoms}")
            print()
        
        # Test fuzzy matching with typos
        print("Testing fuzzy matching with typos:")
        print("-" * 50)
        
        typo_cases = [
            "I have a fevr",  # fever -> fevr
            "My hed hurts",   # head -> hed
            "I have stomch pain",  # stomach -> stomch
            "I feel dizy",    # dizzy -> dizy
            "I have a skin rash with iching",  # itching -> iching
        ]
        
        for test_text in typo_cases:
            symptoms = extractor.extract_symptoms(test_text)
            print(f"• \"{test_text}\"")
            print(f"  Symptoms: {symptoms}")
            print()
        
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        raise
