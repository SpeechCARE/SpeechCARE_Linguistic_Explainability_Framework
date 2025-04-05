import json
from typing import Dict, List, Union
from unsloth import FastLanguageModel
from transformers import TextStreamer, pipeline
import torch
import numpy as np

def format_shap_values(shap_explanation):
    """
    Convert SHAP Explanation object to list of (token, SHAP value) pairs.
    
    Args:
        shap_explanation: SHAP Explanation object
        
    Returns:
        list: List of tuples in format (token, shap_value)
    """
    # Get tokens and values
    tokens = np.array(shap_explanation.data[0])  # Convert to numpy array if not already
    values = shap_explanation.values
        
    # Create (token, value) pairs
    token_value_pairs = []
    for token, value in zip(tokens, values):
        token_str = str(token)
        # Handle scalar values (single classification) or arrays (multi-class)
        shap_value = float(value) if np.isscalar(value) else [float(v) for v in value]
        token_value_pairs.append((token_str, shap_value))
    
    return token_value_pairs

def get_llm_interpretation(transcription: str, shap_values: Union[Dict, List], hf_token: str,predicted_label:int) -> str:
    """
    Analyzes linguistic features and SHAP values to detect cognitive impairment patterns.
    
    Args:
        transcription (str): The text passage to analyze for cognitive impairment cues.
        shap_values (Union[Dict, List]): Token-level SHAP values from a pre-trained model.
        hf_token (str): Hugging Face token
        
    Returns:
        str: The final analysis and prediction regarding cognitive impairment.
    """
    # Initialize model and tokenizer
    model, tokenizer = initialize_model(hf_token)
    
    # Generate initial analysis
    analysis_text = generate_analysis(model, tokenizer, transcription, shap_values)
    
    # Generate final prediction based on analysis
    final_prediction = generate_prediction(model, tokenizer, analysis_text)
    
    return analysis_text, final_prediction

def initialize_model(hf_token):
    """
    Initializes and returns the language model and tokenizer with optimized settings.
    
    Returns:
        tuple: (model, tokenizer) pair for text generation
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.3-70B-Instruct",
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=True,
        token=hf_token,
        cache_dir="/workspace"
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    return model, tokenizer

def generate_analysis(model, tokenizer, transcription: str, shap_values: Union[Dict, List]) -> str:
    """
    Generates the initial linguistic analysis using the provided transcription and SHAP values.
    
    Args:
        model: The language model
        tokenizer: The text tokenizer
        transcription (str): Text to analyze
        shap_values (Union[Dict, List]): SHAP values for interpretation
        
    Returns:
        str: The generated analysis text
    """

    # Prepare the system prompt with the provided inputs
    system_prompt = """
        You are a specialized language model trained to detect linguistic cues of cognitive impairment. You will receive:
        1) A set of linguistic features to consider.
        
        2) A text passage to analyze.
        3) Token-level SHAP values from a pre-trained model.
        ---
        Linguistic Features to Consider:
        • Lexical Richness: Unusual or varied vocabulary, overuse of vague terms (e.g., “thing,” “stuff”).
        • Syntactic Complexity: Simple vs. complex sentence constructions, grammatical errors.
        • Sentence Length and Structure: Fragmented vs. compound/complex sentences.
        • Repetition: Repeated words, phrases, or clauses.
        • Disfluencies and Fillers: Terms like “um,” “uh,” “like.”
        • Semantic Coherence and Content: Logical flow of ideas, clarity of meaning.
        • Additional Feature: Placeholder for any extra marker (e.g., specialized domain terms).
        ---
        Text to Analyze:
        {text}
        ---
        Token-level SHAP Values:
        {shap_values}
        ---
        
        Analysis Format:
        Synthesize the significance of these tokens/features to explain how they collectively point to healthy cognition or potential cognitive impairment.
        Ensure that the explanations are concise, insightful, and relevant to cognitive impairment assessment.
        Output should be structured as **bullet points**, with each bullet clearly describing one key aspect of the analysis. 
        """.format(text=transcription, shap_values=json.dumps(format_shap_values(shap_values), indent=2))

    inputs = tokenizer(system_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Generate text
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            top_p=1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Get only the newly generated tokens (after the input prompt)
    new_tokens = outputs[0][input_ids.shape[1]:]

    # Decode only the new tokens
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text
    

def generate_prediction(model, tokenizer, analysis_text: str) -> str:
    """
    Generates a final prediction based on the initial analysis text.
    
    Args:
        model: The language model
        tokenizer: The text tokenizer
        analysis_text (str): The initial analysis to summarize
        
    Returns:
        str: The final prediction and summary
    """
    system_prompt = f"""
    Based on the analysis of detecting cognitive impairment, provide the following:
    Write the final prediction regarding the detection of cognitive impairment in **one short sentence**.
    Summarize the key findings and their implications in **bullet points**, without using the "Title: description" format.
    ---
    {analysis_text}
    """
    # Tokenize inputs and move to model device
    inputs = tokenizer(system_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,  # Lower for more focused output
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

     # Decode only the newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

 