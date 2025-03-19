import requests
import json
from SHAP.text_visualization import text

def get_llama_interpretation(llama_api_key,transcription,predicted_label,shap_dict, shap_values):
        
        message = f"""
            You are a specialized language model trained to detect linguistic cues of cognitive impairment. You will receive:
            1) A set of linguistic features to consider.
            2) A text passage to analyze.
            3) Token-level SHAP values from a pre-trained model.

            Your task is to:
            A. Identify which tokens are most influential in the classification, based on SHAP values.
            B. Map each influential token to one or more linguistic features (e.g., lexical richness, syntactic complexity).
            C. Explain how the token and its context may indicate healthy cognition or cognitive impairment.

            Please follow the steps below and provide a structured analysis.
            ---
            Linguistic Features to Consider:
            • **Lexical Richness**: Unusual or varied vocabulary, overuse of vague terms (e.g., “thing,” “stuff”).
            • **Syntactic Complexity**: Simple vs. complex sentence constructions, grammatical errors.
            • **Sentence Length and Structure**: Fragmented vs. compound/complex sentences.
            • **Repetition**: Repeated words, phrases, or clauses.
            • **Disfluencies and Fillers**: Terms like “um,” “uh,” “like.”
            • **Semantic Coherence and Content**: Logical flow of ideas, clarity of meaning.
            • **Additional Feature (XXX)**: Placeholder for any extra marker (e.g., specialized domain terms).
            ---
            Text to Analyze:
            {transcription}
            ---
            Token-level SHAP Values:
            {shap_dict}
            ---
            Analysis Format:
            1) **Token-Level Analysis**: For each token with a significant |SHAP| value, specify:
            - Token
            - SHAP Value
            - Linguistic Feature(s) Involved
            - Brief Interpretation
            2) **Overall Summary**: Synthesize the significance of these tokens/features to explain how they collectively point to healthy cognition or potential cognitive impairment.
            ---
            """
        output_format = """
            Example Output Format:
            {
            "Analysis": [
                {
                "Token": "um",
                "SHAP_Value": -0.87,
                "Linguistic_Feature": "Disfluency",
                "Interpretation": "Filler word commonly seen in cognitive impairment contexts."
                },
                {
                "Token": "patient",
                "SHAP_Value": 0.65,
                "Linguistic_Feature": "Lexical Richness",
                "Interpretation": "Use of domain-specific term suggests context awareness."
                },
                ...
            ],
            [sep_token]
            "Overall_Summary": "Multiple disfluencies and repetitive fragments are indicative of possible cognitive impairment."
            [sep_token]
            }
            ---
            Constraints and Guidelines:
            - Rely only on the provided text and SHAP values; do not infer from external or hidden knowledge.
            - Tie each token with high |SHAP| back to a specific linguistic feature and explain its clinical relevance.
            - Put separator token [sep_token] before and after the 'Overall_Summary' in the output.
            """

        message += output_format

        print(f"Getting LLaMA interpretation with api key: {llama_api_key}")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {llama_api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [
                {
                    "role": "user",
                    "content": message
                }
                ],
            })
            )
        print(response.json())
        shap_html_code = text(shap_values[:,:,predicted_label], display=False)
        return shap_html_code, response.json()['choices'][0]['message']['content']


