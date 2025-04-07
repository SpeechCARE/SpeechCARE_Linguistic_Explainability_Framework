import json
from typing import Dict, List, Union
from unsloth import FastLanguageModel
import torch
import numpy as np
from modelscope import AutoTokenizer
from modelscope import AutoModelForCausalLM

system_prompt1 = """
    You are a specialized language model trained to detect linguistic cues of cognitive impairment. You will receive:
    1) A set of linguistic features to consider.
    2) A text passage to analyze.
    3) Token-level SHAP values from a pre-trained model.
    
    ---
    ## Linguistic Features to Consider:
    • Lexical Richness: Unusual or varied vocabulary, overuse of vague terms (e.g., “thing,” “stuff”).
    • Syntactic Complexity: Simple vs. complex sentence constructions, grammatical errors.
    • Sentence Length and Structure: Fragmented vs. compound/complex sentences.
    • Repetition: Repeated words, phrases, or clauses.
    • Disfluencies and Fillers: Terms like “um,” “uh,” “like.”
    • Semantic Coherence and Content: Logical flow of ideas, clarity of meaning.
    • Additional Feature: Placeholder for any extra marker (e.g., specialized domain terms).
    ---
    ## Text to Analyze:
    {text}
    ---
    ## Token-level SHAP Values:
    {shap_values}
    ---
    You must analyze the given text and the shap values based on:
    Synthesize the significance of provided tokens/features to explain how they collectively point to healthy cognition or potential cognitive impairment.
    Ensure that the explanations are concise, insightful, and relevant to cognitive impairment assessment.
    Output should be structured as **bullet points**, with each bullet clearly describing one key aspect of the analysis. 
    ---
    ## Analysis:
    
    """

system_prompt2 = """
    Based on the the long analysis of detecting cognitive impairment, provide the following:
    Write the final prediction regarding the detection of cognitive impairment in **one short sentence**.
    Summarize the key findings and their implications in **bullet points**, without using the "Title: description" format.
    Do not provide any additional or extra explanations and points.
    **Avoid saying anything about SHAP values and Giving Suggestions for further analysis**
    **Do not repeat yourself**
    ---
    ## Long Analysis:
    {generated_text}

    ---
    ## Final Prediction and Key findingd:
    
    """
def get_pairs(tokens, shaps, index):
    list_of_pairs = []
    for token, shap in zip(tokens, shaps):
        list_of_pairs.append([token, shap[index] if shap[index] > 0 else None])
    return list_of_pairs



def apply_chat_template(template,tokenizer):
    return tokenizer.apply_chat_template(template, tokenize=False)

def prep_prompt_analysis(transcript, shap_values,tokenizer):
    content = system_prompt1.format(text=transcript, shap_values=json.dumps(shap_values, indent=2))
    messages = [{"role": "user", "content": content}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

def prep_prompt_summarize(generated_text,tokenizer):
    content = system_prompt2.format(generated_text=generated_text)
    messages = [{"role": "user", "content": content}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

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

def get_llm_interpretation(transcription: str, shap_values: Union[Dict, List],shap_index) -> str:
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
    model, tokenizer = initialize_model()
    
    # Generate initial analysis
    analysis_text = generate_analysis(model, tokenizer, transcription, shap_values,shap_index)
    
    # Generate final prediction based on analysis
    final_prediction = generate_prediction(model, tokenizer, analysis_text)
    
    return analysis_text, final_prediction

def initialize_model(model_path ='/workspace/models/llama70B'):
    """
    Initializes and returns the language model and tokenizer with optimized settings.
    
    Returns:
        tuple: (model, tokenizer) pair for text generation
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
   
    return model, tokenizer

def generate_analysis(model, tokenizer, transcription: str, shap_values: Union[Dict, List],shap_index:int) -> str:
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
    token_shap_pairs = get_pairs(shap_values.tokens, shap_values.shap_values, shap_index)

    prompt = system_prompt1.format(text=transcription, shap_values=json.dumps(token_shap_pairs, indent=2))
    inputs1 = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids1 = inputs1["input_ids"]

    # Generate text
    with torch.inference_mode():
        outputs1 = model.generate(
            **inputs1,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            top_p=1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Get only the newly generated tokens (after the input prompt)
    new_tokens1 = outputs1[0][input_ids1.shape[1]:]

    # Decode only the new tokens
    generated_text1 = tokenizer.decode(new_tokens1, skip_special_tokens=True)
    return generated_text1
    

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
    prompt2 = prep_prompt_summarize(analysis_text)

    # Tokenize
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
    input_ids2 = inputs2["input_ids"]

    # Generate text
    with torch.inference_mode():
        outputs2 = model.generate(
            **inputs2,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.1,
            top_p=1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Get only the newly generated tokens (after the input prompt)
    new_tokens2 = outputs2[0][input_ids2.shape[1]:]

    # Decode only the new tokens
    generated_text2 = tokenizer.decode(new_tokens2, skip_special_tokens=True)
    return generated_text2

 