"""
Translation module for symptoms checker project.
Provides translation services with Google Cloud Translate and fallback options.
"""

import os
import re
from typing import Optional, Dict, Any
from pathlib import Path


class Translator:
    """
    Handles text translation between languages with multiple backend options.
    
    Production Mode: Uses Google Cloud Translate (requires GOOGLE_APPLICATION_CREDENTIALS)
    Prototyping Mode: Uses googletrans library (free but rate-limited)
    """
    
    def __init__(self):
        """
        Initialize translator with appropriate backend based on environment.
        """
        self.google_cloud_available = self._check_google_cloud_credentials()
        self.googletrans_available = self._check_googletrans_availability()
        self.deep_translator_available = self._check_deep_translator_availability()
        
        # Initialize appropriate client
        self.client = None
        self.translation_mode = None
        
        if self.google_cloud_available:
            self._init_google_cloud_client()
            self.translation_mode = "google_cloud"
            print("✓ Using Google Cloud Translate (Production Mode)")
        elif self.googletrans_available:
            self._init_googletrans_client()
            self.translation_mode = "googletrans"
            print("✓ Using googletrans library (Prototyping Mode)")
        elif self.deep_translator_available:
            self._init_deep_translator_client()
            self.translation_mode = "deep_translator"
            print("✓ Using deep-translator library (Fallback Mode)")
        else:
            print("⚠️ No translation service available. Install google-cloud-translate, googletrans, or deep-translator")
            self.translation_mode = "none"
    
    def _check_google_cloud_credentials(self) -> bool:
        """
        Check if Google Cloud credentials are available.
        
        Returns:
            True if GOOGLE_APPLICATION_CREDENTIALS environment variable is set
        """
        return os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is not None
    
    def _check_googletrans_availability(self) -> bool:
        """
        Check if googletrans library is available.
        
        Returns:
            True if googletrans can be imported
        """
        try:
            import googletrans
            return True
        except (ImportError, ModuleNotFoundError):
            return False
    
    def _check_deep_translator_availability(self) -> bool:
        """
        Check if deep-translator library is available.
        
        Returns:
            True if deep-translator can be imported
        """
        try:
            from deep_translator import GoogleTranslator
            return True
        except (ImportError, ModuleNotFoundError):
            return False
    
    def _init_google_cloud_client(self) -> None:
        """
        Initialize Google Cloud Translate client.
        This is the PRODUCTION mode - requires valid Google Cloud credentials.
        """
        try:
            from google.cloud import translate_v2 as translate
            self.client = translate.Client()
            print("✓ Google Cloud Translate client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Google Cloud Translate: {e}")
            self.client = None
    
    def _init_googletrans_client(self) -> None:
        """
        Initialize googletrans client.
        This is the PROTOTYPING mode - free but has rate limits and reliability issues.
        """
        try:
            from googletrans import Translator as GoogletransTranslator
            self.client = GoogletransTranslator()
            print("✓ googletrans client initialized")
        except (ImportError, ModuleNotFoundError, Exception) as e:
            print(f"❌ Failed to initialize googletrans: {e}")
            print("Note: googletrans may have compatibility issues with Python 3.13+")
            self.client = None
    
    def _init_deep_translator_client(self) -> None:
        """
        Initialize deep-translator client.
        This is the FALLBACK mode - free and more reliable than googletrans.
        """
        try:
            from deep_translator import GoogleTranslator
            self.client = GoogleTranslator
            print("✓ deep-translator client initialized")
        except (ImportError, ModuleNotFoundError, Exception) as e:
            print(f"❌ Failed to initialize deep-translator: {e}")
            self.client = None
    
    def translate_to_english(self, text: str, src: str = 'auto') -> str:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            src: Source language code ('auto' for auto-detection)
            
        Returns:
            Translated text in English
            
        Raises:
            ValueError: If translation fails or no client available
        """
        if not text or not text.strip():
            return text
        
        if self.translation_mode == "none":
            raise ValueError("No translation service available")
        
        if self.translation_mode == "google_cloud":
            return self._translate_google_cloud(text, src, 'en')
        elif self.translation_mode == "googletrans":
            return self._translate_googletrans(text, src, 'en')
        elif self.translation_mode == "deep_translator":
            return self._translate_deep_translator(text, src, 'en')
        else:
            raise ValueError(f"Unknown translation mode: {self.translation_mode}")
    
    def translate_from_english(self, text: str, target: str) -> str:
        """
        Translate text from English to target language.
        
        Args:
            text: English text to translate
            target: Target language code (e.g., 'hi', 'pa', 'es')
            
        Returns:
            Translated text in target language
            
        Raises:
            ValueError: If translation fails or no client available
        """
        if not text or not text.strip():
            return text
        
        if self.translation_mode == "none":
            raise ValueError("No translation service available")
        
        if self.translation_mode == "google_cloud":
            return self._translate_google_cloud(text, 'en', target)
        elif self.translation_mode == "googletrans":
            return self._translate_googletrans(text, 'en', target)
        elif self.translation_mode == "deep_translator":
            return self._translate_deep_translator(text, 'en', target)
        else:
            raise ValueError(f"Unknown translation mode: {self.translation_mode}")
    
    def _translate_google_cloud(self, text: str, src: str, target: str) -> str:
        """
        Translate using Google Cloud Translate (Production Mode).
        
        Advantages:
        - High reliability and accuracy
        - No rate limits for paid accounts
        - Professional API with SLA
        - Supports 100+ languages
        
        Args:
            text: Text to translate
            src: Source language code
            target: Target language code
            
        Returns:
            Translated text
        """
        try:
            result = self.client.translate(text, source_language=src, target_language=target)
            return result['translatedText']
        except Exception as e:
            raise ValueError(f"Google Cloud translation failed: {e}")
    
    def _translate_googletrans(self, text: str, src: str, target: str) -> str:
        """
        Translate using googletrans library (Prototyping Mode).
        
        Limitations:
        - Rate limited (may fail with frequent requests)
        - Less reliable than Google Cloud API
        - May have connection issues
        - Free but unofficial library
        
        Args:
            text: Text to translate
            src: Source language code
            target: Target language code
            
        Returns:
            Translated text
        """
        try:
            result = self.client.translate(text, src=src, dest=target)
            return result.text
        except Exception as e:
            raise ValueError(f"googletrans translation failed: {e}")
    
    def _translate_deep_translator(self, text: str, src: str, target: str) -> str:
        """
        Translate using deep-translator library (Fallback Mode).
        
        Advantages:
        - More reliable than googletrans
        - Better Python 3.13+ compatibility
        - Free and open source
        - Good fallback option
        
        Args:
            text: Text to translate
            src: Source language code
            target: Target language code
            
        Returns:
            Translated text
        """
        try:
            translator = self.client(source=src, target=target)
            result = translator.translate(text)
            return result
        except Exception as e:
            raise ValueError(f"deep-translator translation failed: {e}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text using heuristic methods.
        
        Heuristic Detection:
        - Checks for Devanagari script characters (Hindi)
        - Checks for Gurmukhi script characters (Punjabi)
        - Falls back to 'en' (English) if no script detected
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code: 'en', 'hi', 'pa', or 'en' (default)
        """
        if not text or not text.strip():
            return 'en'
        
        # Check for Devanagari script (Hindi)
        # Devanagari Unicode range: U+0900-U+097F
        devanagari_pattern = r'[\u0900-\u097F]'
        if re.search(devanagari_pattern, text):
            return 'hi'
        
        # Check for Gurmukhi script (Punjabi)
        # Gurmukhi Unicode range: U+0A00-U+0A7F
        gurmukhi_pattern = r'[\u0A00-\u0A7F]'
        if re.search(gurmukhi_pattern, text):
            return 'pa'
        
        # Default to English if no script detected
        return 'en'
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages based on current translation mode.
        
        Returns:
            Dictionary mapping language codes to language names
        """
        if self.translation_mode == "google_cloud":
            return self._get_google_cloud_languages()
        elif self.translation_mode == "googletrans":
            return self._get_googletrans_languages()
        else:
            return {'en': 'English'}
    
    def _get_google_cloud_languages(self) -> Dict[str, str]:
        """
        Get supported languages from Google Cloud Translate.
        
        Returns:
            Dictionary of language codes and names
        """
        try:
            languages = self.client.get_languages()
            return {lang['language']: lang['name'] for lang in languages}
        except Exception:
            # Fallback to common languages
            return {
                'en': 'English',
                'hi': 'Hindi',
                'pa': 'Punjabi',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh': 'Chinese'
            }
    
    def _get_googletrans_languages(self) -> Dict[str, str]:
        """
        Get supported languages from googletrans.
        
        Returns:
            Dictionary of language codes and names
        """
        try:
            # googletrans has a LANGUAGES constant
            from googletrans import LANGUAGES
            return LANGUAGES
        except Exception:
            # Fallback to common languages
            return {
                'en': 'English',
                'hi': 'Hindi',
                'pa': 'Punjabi',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh': 'Chinese'
            }
    
    def get_translation_info(self) -> Dict[str, Any]:
        """
        Get information about current translation setup.
        
        Returns:
            Dictionary containing translation mode and capabilities
        """
        return {
            'mode': self.translation_mode,
            'google_cloud_available': self.google_cloud_available,
            'googletrans_available': self.googletrans_available,
            'client_initialized': self.client is not None,
            'supported_languages': len(self.get_supported_languages())
        }


# Global translator instance
_translator_instance = None


def get_translator() -> Translator:
    """
    Get or create global translator instance.
    
    Returns:
        Translator instance
    """
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = Translator()
    return _translator_instance


def translate_to_english(text: str, src: str = 'auto') -> str:
    """
    Convenience function to translate text to English.
    
    Args:
        text: Text to translate
        src: Source language code
        
    Returns:
        Translated text in English
    """
    translator = get_translator()
    return translator.translate_to_english(text, src)


def translate_from_english(text: str, target: str) -> str:
    """
    Convenience function to translate text from English.
    
    Args:
        text: English text to translate
        target: Target language code
        
    Returns:
        Translated text in target language
    """
    translator = get_translator()
    return translator.translate_from_english(text, target)


def detect_language(text: str) -> str:
    """
    Convenience function to detect language.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code
    """
    translator = get_translator()
    return translator.detect_language(text)


if __name__ == "__main__":
    """
    Test examples demonstrating translation functionality.
    """
    print("Testing Translation Module...")
    print("=" * 50)
    
    try:
        translator = get_translator()
        
        # Display translation info
        info = translator.get_translation_info()
        print(f"Translation Mode: {info['mode']}")
        print(f"Google Cloud Available: {info['google_cloud_available']}")
        print(f"googletrans Available: {info['googletrans_available']}")
        print(f"Client Initialized: {info['client_initialized']}")
        print(f"Supported Languages: {info['supported_languages']}")
        
        # Test language detection
        print(f"\nTesting Language Detection:")
        print("-" * 30)
        
        test_texts = [
            "I have a fever and headache",
            "मुझे बुखार और सिरदर्द है",  # Hindi
            "ਮੈਨੂੰ ਬੁਖਾਰ ਅਤੇ ਸਿਰਦਰਦ ਹੈ",  # Punjabi
            "J'ai de la fièvre et mal à la tête",  # French
            "Tengo fiebre y dolor de cabeza"  # Spanish
        ]
        
        for text in test_texts:
            detected = translator.detect_language(text)
            print(f"Text: \"{text[:30]}...\"")
            print(f"Detected: {detected}")
            print()
        
        # Test translation (if client is available)
        if translator.client is not None:
            print(f"Testing Translation:")
            print("-" * 20)
            
            test_cases = [
                ("I have a fever", "hi"),  # English to Hindi
                ("I have a headache", "pa"),  # English to Punjabi
                ("I feel dizzy", "es"),  # English to Spanish
            ]
            
            for english_text, target_lang in test_cases:
                try:
                    translated = translator.translate_from_english(english_text, target_lang)
                    print(f"English: \"{english_text}\"")
                    print(f"Target: {target_lang}")
                    print(f"Translated: \"{translated}\"")
                    print()
                except Exception as e:
                    print(f"Translation failed for {target_lang}: {e}")
                    print()
        else:
            print("⚠️ No translation client available - skipping translation tests")
        
        # Show supported languages
        print(f"Supported Languages:")
        print("-" * 20)
        languages = translator.get_supported_languages()
        common_languages = ['en', 'hi', 'pa', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        
        for lang_code in common_languages:
            if lang_code in languages:
                print(f"{lang_code}: {languages[lang_code]}")
        
        print(f"\n✓ Translation module testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        raise
