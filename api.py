"""
FastAPI application for symptoms checker project.

Sample curl request:
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "input": "I have a fever and headache",
       "input_type": "text",
       "language": "en",
       "mode": "text"
     }'

For audio input:
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "input": "base64_encoded_audio_data",
       "input_type": "audio",
       "language": "hi",
       "mode": "voice"
     }'
"""

import base64
import json
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our custom modules
from translator import translate_to_english, translate_from_english, detect_language
from symptom_extractor import extract_symptoms
from feature_engineer import build_feature_vector
from stt_tts import speech_to_text_cloud, speech_to_text_local, text_to_speech_cloud, text_to_speech_local
from disease_matcher import DiseaseMatcher

# Configuration flags
USE_SCORER_ONLY = False  # Set to True to use only rule-based scoring, False for ensemble

# Pydantic models for request/response
class PredictRequest(BaseModel):
    input: str = Field(..., description="User text or base64-encoded audio data")
    input_type: str = Field(..., description="Type of input: 'text' or 'audio'")
    language: str = Field(..., description="Language code: 'en', 'hi', or 'pa'")
    mode: str = Field(..., description="Output mode: 'text' or 'voice'")


class DiseasePrediction(BaseModel):
    disease: str
    prob: float
    severity: str
    precautions: List[str]
    symptom_descriptions: Dict[str, str]


class PredictResponse(BaseModel):
    input_text: str
    input_text_user_lang: str
    symptoms: List[str]
    predictions: List[DiseasePrediction]
    predictions_translated: Optional[List[DiseasePrediction]] = None
    language: str
    display_text: str
    tts_audio_base64: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


# Global variables for loaded artifacts
model_data = None
meta_data = None
symptom_severity = None
disease_precautions = None
symptom_descriptions = None
disease_matcher = None


def load_artifacts():
    """
    Load all required artifacts on startup.
    """
    global model_data, meta_data, symptom_severity, disease_precautions, symptom_descriptions, disease_matcher
    
    artifacts_dir = Path("artifacts")
    
    try:
        # Load trained model
        model_path = artifacts_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model_data = joblib.load(model_path)
        print(f"✓ Loaded model from {model_path}")
        
        # Load metadata
        meta_path = artifacts_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta data not found: {meta_path}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        print(f"✓ Loaded metadata from {meta_path}")
        
        # Load symptom severity mapping
        severity_path = artifacts_dir / "symptom_severity.json"
        if not severity_path.exists():
            raise FileNotFoundError(f"Symptom severity not found: {severity_path}")
        with open(severity_path, 'r', encoding='utf-8') as f:
            symptom_severity = json.load(f)
        print(f"✓ Loaded symptom severity from {severity_path}")
        
        # Load disease precautions mapping
        precautions_path = artifacts_dir / "disease_precaution_map.json"
        if not precautions_path.exists():
            raise FileNotFoundError(f"Disease precautions not found: {precautions_path}")
        with open(precautions_path, 'r', encoding='utf-8') as f:
            disease_precautions = json.load(f)
        print(f"✓ Loaded disease precautions from {precautions_path}")
        
        # Load symptom descriptions
        descriptions_path = artifacts_dir / "symptom_description.json"
        if not descriptions_path.exists():
            print(f"⚠️ Symptom descriptions not found: {descriptions_path}")
            symptom_descriptions = {}
        else:
            with open(descriptions_path, 'r', encoding='utf-8') as f:
                symptom_descriptions = json.load(f)
            print(f"✓ Loaded symptom descriptions from {descriptions_path}")
        
        # Initialize disease matcher
        try:
            disease_matcher = DiseaseMatcher(str(artifacts_dir))
            print(f"✓ Disease matcher initialized successfully")
        except Exception as e:
            print(f"⚠️ Disease matcher initialization failed: {e}")
            disease_matcher = None
        
        print(f"✓ All artifacts loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise


def process_audio_input(audio_base64: str, language_code: str) -> str:
    """
    Process audio input and return transcribed text.
    
    Args:
        audio_base64: Base64-encoded audio data
        language_code: Language code for transcription
        
    Returns:
        Transcribed text in English
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Try cloud STT first, fallback to local
        try:
            # Convert language code for Google Cloud (e.g., 'hi' -> 'hi-IN')
            cloud_lang_code = f"{language_code}-IN" if language_code in ['hi', 'pa'] else f"{language_code}-US"
            transcribed_text = speech_to_text_cloud(audio_bytes, cloud_lang_code)
        except Exception:
            # Fallback to local STT (requires saving to temp file)
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            try:
                transcribed_text = speech_to_text_local(temp_file_path)
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        return transcribed_text.strip()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")


def calculate_disease_severity(symptoms: List[str], disease: str) -> str:
    """
    Calculate disease severity based on symptom weights.
    
    Args:
        symptoms: List of symptoms for the disease
        disease: Disease name
        
    Returns:
        Severity label: 'Low', 'Medium', 'High', or 'Critical'
    """
    if not symptoms or not symptom_severity:
        return "Low"
    
    # Get weights for symptoms
    weights = []
    for symptom in symptoms:
        if symptom in symptom_severity:
            weights.append(symptom_severity[symptom])
    
    if not weights:
        return "Low"
    
    # Use max aggregation (most severe symptom determines overall severity)
    max_weight = max(weights)
    
    # Map to severity labels using thresholds from meta.json
    severity_thresholds = meta_data.get('severity_thresholds', {
        'low': 0.0,
        'medium': 0.3,
        'high': 0.6,
        'critical': 0.8
    })
    
    if max_weight >= severity_thresholds['critical']:
        return "Critical"
    elif max_weight >= severity_thresholds['high']:
        return "High"
    elif max_weight >= severity_thresholds['medium']:
        return "Medium"
    else:
        return "Low"


def generate_display_text(predictions: List[DiseasePrediction], language: str) -> str:
    """
    Generate natural language summary of predictions.
    
    Args:
        predictions: List of disease predictions
        language: Target language for translation
        
    Returns:
        Translated display text
    """
    if not predictions:
        return "No diseases predicted based on your symptoms."
    
    # Create English summary
    if len(predictions) == 1:
        disease = predictions[0].disease
        prob = predictions[0].prob
        severity = predictions[0].severity
        english_text = f"Based on your symptoms, you may have {disease} (probability: {prob:.1%}, severity: {severity})."
    else:
        top_disease = predictions[0].disease
        prob = predictions[0].prob
        severity = predictions[0].severity
        english_text = f"Based on your symptoms, the most likely condition is {top_disease} (probability: {prob:.1%}, severity: {severity})."
    
    # Translate to user's language
    try:
        if language != 'en':
            translated_text = translate_from_english(english_text, language)
            return translated_text
        else:
            return english_text
    except Exception:
        # Fallback to English if translation fails
        return english_text


# Create FastAPI app
app = FastAPI(
    title="Symptoms Checker API",
    description="AI-powered symptom analysis and disease prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Load artifacts on application startup.
    """
    print("Starting Symptoms Checker API...")
    load_artifacts()
    print("✓ API startup completed")


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Symptoms Checker API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "artifacts_loaded": all([
            model_data is not None,
            meta_data is not None,
            symptom_severity is not None,
            disease_precautions is not None
        ])
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Main prediction endpoint that processes text or audio input and returns disease predictions.
    
    Args:
        request: Prediction request with input, type, language, and mode
        
    Returns:
        Prediction response with diseases, probabilities, and additional information
    """
    try:
        # Validate input
        if request.input_type not in ['text', 'audio']:
            raise HTTPException(status_code=400, detail="input_type must be 'text' or 'audio'")
        
        if request.language not in ['en', 'hi', 'pa']:
            raise HTTPException(status_code=400, detail="language must be 'en', 'hi', or 'pa'")
        
        if request.mode not in ['text', 'voice']:
            raise HTTPException(status_code=400, detail="mode must be 'text' or 'voice'")
        
        # Step 1: Process input based on type
        if request.input_type == 'audio':
            # Process audio input
            input_text = process_audio_input(request.input, request.language)
            input_text_user_lang = input_text  # Keep original transcript
        else:
            # Use text input directly
            input_text = request.input
            input_text_user_lang = request.input
        
        if not input_text.strip():
            raise HTTPException(status_code=400, detail="No text found in input")
        
        # Debug logging: Original user text and language
        print(f"🔍 Original user text: '{input_text_user_lang}'")
        print(f"🌐 Selected language: '{request.language}'")
        
        # Step 2: Force translation to English if needed
        if request.language != 'en':
            try:
                print(f"🔄 Translating from {request.language} to English...")
                input_text = translate_to_english(input_text_user_lang, request.language)
                print(f"✅ Translated English text: '{input_text}'")
            except Exception as e:
                print(f"❌ Translation error: {e}")
                print(f"⚠️ Continuing with original text: '{input_text}'")
                # Continue with original text if translation fails
        else:
            print(f"✅ Text already in English: '{input_text}'")
        
        # Step 3: Extract symptoms
        symptoms = extract_symptoms(input_text)
        print(f"🎯 Extracted symptoms: {symptoms}")
        
        if not symptoms:
            # Return empty prediction if no symptoms found
            return PredictResponse(
                input_text=input_text,
                input_text_user_lang=input_text_user_lang,
                symptoms=[],
                predictions=[],
                language=request.language,
                display_text=generate_display_text([], request.language),
                tts_audio_base64=None
            )
        
        # Step 4: Build feature vector and predict
        feature_vector = build_feature_vector(symptoms, str(Path("artifacts")))
        feature_vector = feature_vector.reshape(1, -1)
        
        # Get ML model probabilities (if available and not scorer-only mode)
        model_probs = None
        if not USE_SCORER_ONLY and model_data and 'model' in model_data:
            try:
                model_probs = model_data['model'].predict_proba(feature_vector)[0]
                print(f"🤖 ML model predictions: {len(model_probs)} diseases")
            except Exception as e:
                print(f"⚠️ ML model prediction failed: {e}")
                model_probs = None
        
        # Run rule-based disease scoring
        diseases = meta_data['diseases']
        scorer_results = []
        scorer_probs = []
        
        if disease_matcher:
            try:
                scorer_results = disease_matcher.score_diseases(symptoms, top_k=len(diseases))
                print(f"📊 Rule-based scoring: {len(scorer_results)} diseases scored")
                
                # Create scorer probabilities ordered same as meta_data['diseases']
                disease_to_prob = {result['disease']: result['prob'] for result in scorer_results}
                scorer_probs = [disease_to_prob.get(disease, 0.0) for disease in diseases]
                
                # Log top 3 scorer predictions
                top_scorer = scorer_results[:3]
                scorer_log = ", ".join([f"{r['disease']}:{r['prob']:.3f}" for r in top_scorer])
                print(f"🎯 Top 3 scorer predictions: {scorer_log}")
                
            except Exception as e:
                print(f"⚠️ Rule-based scoring failed: {e}")
                scorer_results = []
                scorer_probs = []
        
        # Combine ML and rule-based predictions
        if model_probs is not None and scorer_probs and not USE_SCORER_ONLY:
            try:
                final_probs = disease_matcher.ensemble_with_model(model_probs, scorer_probs, model_weight=0.35)
                print(f"🔄 Ensemble predictions: ML weight 0.35, Scorer weight 0.65")
            except Exception as e:
                print(f"⚠️ Ensemble failed, using scorer only: {e}")
                final_probs = scorer_probs
        else:
            final_probs = scorer_probs if scorer_probs else model_probs
            if final_probs is None:
                print("❌ No predictions available")
                return PredictResponse(
                    input_text=input_text,
                    input_text_user_lang=input_text_user_lang,
                    symptoms=symptoms,
                    predictions=[],
                    predictions_translated=None,
                    language=request.language,
                    display_text=generate_display_text([], request.language),
                    tts_audio_base64=None
                )
        
        # Step 5: Create disease predictions
        threshold = meta_data.get('threshold', 0.3)
        disease_predictions = []
        
        for i, (disease, prob) in enumerate(zip(diseases, final_probs)):
            if prob >= threshold or i < 3:  # Always include top 3, others above threshold
                # Get precautions for this disease
                precautions = disease_precautions.get(disease, [])
                
                # Get symptom descriptions for this disease's symptoms
                symptom_descriptions_dict = {}
                for symptom in symptoms:
                    if symptom in symptom_descriptions:
                        symptom_descriptions_dict[symptom] = symptom_descriptions[symptom]
                
                # Calculate severity
                severity = calculate_disease_severity(symptoms, disease)
                
                disease_predictions.append(DiseasePrediction(
                    disease=disease,
                    prob=float(prob),
                    severity=severity,
                    precautions=precautions,
                    symptom_descriptions=symptom_descriptions_dict
                ))
        
        # Sort by probability (descending)
        disease_predictions.sort(key=lambda x: x.prob, reverse=True)
        
        # Take top 3 for UI ranking
        disease_predictions = disease_predictions[:3]
        
        # Log top 3 final predictions
        top_final = disease_predictions[:3]
        final_log = ", ".join([f"{p.disease}:{p.prob:.3f}" for p in top_final])
        print(f"🏆 Top 3 final predictions: {final_log}")
        
        # Create debug information
        debug_info = {}
        if model_probs is not None:
            top_model_indices = np.argsort(model_probs)[-3:][::-1]
            debug_info["model_probs_top3"] = [
                {"disease": diseases[i], "prob": float(model_probs[i])} 
                for i in top_model_indices
            ]
        if scorer_results:
            debug_info["scorer_top3"] = [
                {"disease": r["disease"], "prob": r["prob"], "matched": r["matched"]} 
                for r in scorer_results[:3]
            ]
        
        # Step 6: Translate predictions if needed
        predictions_translated = None
        if request.language != 'en':
            try:
                print(f"🔄 Translating predictions to {request.language}...")
                predictions_translated = []
                for prediction in disease_predictions:
                    # Convert Pydantic model to dict for translation
                    prediction_dict = prediction.dict()
                    
                    # Translate disease name
                    try:
                        prediction_dict["disease"] = translate_from_english(
                            prediction_dict["disease"], request.language
                        )
                        print(f"✅ Translated disease: '{prediction.disease}' → '{prediction_dict['disease']}'")
                    except Exception as e:
                        print(f"⚠️ Disease translation failed: {e}")
                        prediction_dict["disease"] = prediction.disease
                    
                    # Translate precautions
                    try:
                        translated_precautions = []
                        for precaution in prediction_dict["precautions"]:
                            translated_precautions.append(
                                translate_from_english(precaution, request.language)
                            )
                        prediction_dict["precautions"] = translated_precautions
                        print(f"✅ Translated precautions: {len(translated_precautions)} items")
                    except Exception as e:
                        print(f"⚠️ Precautions translation failed: {e}")
                        prediction_dict["precautions"] = prediction.precautions
                    
                    # Convert back to DiseasePrediction model
                    translated_prediction = DiseasePrediction(**prediction_dict)
                    predictions_translated.append(translated_prediction)
                
                print(f"✅ Successfully translated {len(predictions_translated)} predictions")
            except Exception as e:
                print(f"❌ Prediction translation failed: {e}")
                predictions_translated = None
        
        # Step 7: Generate display text
        display_text = generate_display_text(disease_predictions, request.language)
        
        # Step 7: Generate TTS audio if voice mode
        tts_audio_base64 = None
        if request.mode == 'voice':
            try:
                # Try cloud TTS first, fallback to local
                try:
                    cloud_lang_code = f"{request.language}-IN" if request.language in ['hi', 'pa'] else f"{request.language}-US"
                    _, audio_bytes = text_to_speech_cloud(display_text, cloud_lang_code)
                except Exception:
                    # Fallback to local TTS
                    _, audio_bytes, temp_path = text_to_speech_local(display_text, request.language)
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                
                # Encode audio as base64
                tts_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
            except Exception as e:
                print(f"TTS warning: {e}")
                # Continue without audio if TTS fails
        
        # Return response
        return PredictResponse(
            input_text=input_text,
            input_text_user_lang=input_text_user_lang,
            symptoms=symptoms,
            predictions=disease_predictions,
            predictions_translated=predictions_translated,
            language=request.language,
            display_text=display_text,
            tts_audio_base64=tts_audio_base64,
            debug=debug_info if debug_info else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def find_available_port(start_port=8000, max_port=8010):
    """
    Find an available port starting from start_port up to max_port.
    
    Args:
        start_port: Starting port number
        max_port: Maximum port number to try
        
    Returns:
        Available port number
        
    Raises:
        RuntimeError: If no available port found
    """
    import socket
    
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"No available port found between {start_port} and {max_port}")


if __name__ == "__main__":
    import uvicorn
    
    # Find available port
    try:
        port = find_available_port()
        print(f"⚡ API running on http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print(f"❤️ Health Check: http://localhost:{port}/health")
        print("=" * 50)
        
        uvicorn.run(app, host="0.0.0.0", port=port)
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("Please free up some ports or modify the port range in the code.")
