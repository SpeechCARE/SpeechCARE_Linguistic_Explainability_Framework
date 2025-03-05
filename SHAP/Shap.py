import sys
import torch
import shap
import numpy as np

# Add custom paths to sys.path (if needed)
sys.path.append("")
from SpeechCARE_Linguistic_Explainability_Framework.SHAP.text_visualization import text, unpack_shap_explanation_contents, process_shap_values


class LinguisticShap():
    def __init__(self,model):
        self.model = model
        self.text_explainer = shap.Explainer(self.calculate_text_shap_values, self.model.tokenizer, output_names=self.model.labels, hierarchical_values=True)

    def calculate_text_shap_values(self, text):
        device = next(self.model.parameters()).device
        # Tokenize and encode the input
        input_ids = torch.tensor([self.model.tokenizer.encode(v, padding="max_length", max_length=300, truncation=True) for v in text]).to(device)
        attention_masks = (input_ids != 0).type(torch.int64).to(device)
        # Pass through the model
        # outputs = self.text_only_classification(input_ids, attention_masks).detach().cpu().numpy()
        txt_embeddings = self.model.txt_transformer(input_ids=input_ids, attention_mask=attention_masks)
        txt_cls = txt_embeddings.last_hidden_state[:, 0, :]
        txt_x = self.model.txt_head(txt_cls)  
        txt_out = self.model.txt_classifier(txt_x)
        outputs = txt_out.detach().cpu().numpy()

        # Apply softmax to get probabilities
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T

        # Define a helper function to calculate logit with special handling
        def safe_logit(p):
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for divide by zero or invalid ops
                logit = np.log(p / (1 - p))
                logit[p == 0] = -np.inf  # logit(0) = -inf
                logit[p == 1] = np.inf   # logit(1) = inf
                logit[(p < 0) | (p > 1)] = np.nan  # logit(p) is nan for p < 0 or p > 1
            return logit

        # Calculate the new scores based on the specified criteria
        p_0, p_1, p_2 = scores[:, 0], scores[:, 1], scores[:, 2]

        score_0 = safe_logit(p_0)
        p_1_p_2_sum = p_1 + p_2
        score_1 = safe_logit(p_1_p_2_sum)
        score_2 = score_1  # Same as score_1 per your criteria

        # Combine the scores into a single array
        new_scores = np.stack([score_0, score_1, score_2], axis=-1)

        return new_scores
    
    def get_text_shap_results(self, file_name= None):
        """
        Generate SHAP values for the model's transcription and save the results as an HTML file.

        Args:
            file_name (str): The name of the file to save the HTML output.

        Returns:
            str: The generated HTML code.
        """
        print('Running SHAP values...')
        
        # Prepare input text
        input_text = [str(self.model.transcription)]
        print('Input text:', input_text)
        
        # Compute SHAP values
        shap_values = self.text_explainer(input_text)
        print('Values explained...')
        
        # Generate HTML code
        shap_html_code = text(shap_values[:, :, self.model.predicted_label], display=False)
        
        if file_name:
            # Save HTML code to file
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(shap_html_code)
            print(f"SHAP results saved to {file_name}")
            
        return shap_html_code

    def get_text_shap_dict(self, grouping_threshold=0.01, separator=" "):
        """
        Returns a dictionary of token -> SHAP value for the model's current transcription
        and predicted label.
    
        Parameters
        ----------
        grouping_threshold : float
            Merges tokens into a single 'group' if the interaction level is high compared
            to this threshold. A value near 0.01 is common for minimal grouping.
        separator : str
            Used to join subwords if they are merged due to hierarchical grouping. Typically
            ' ' for normal words, or '' if using GPT-style 'Ġ' tokens.
    
        Returns
        -------
        dict
            A dictionary mapping each (possibly merged) token to its SHAP value.
        """
    
        # If there's no transcription yet, run inference or make sure self.transcription is set
        if not self.transcription:
            raise ValueError("self.transcription is empty. Run inference(...) first or set it manually.")
    
        # Get SHAP values for the text (this uses calculate_text_shap_values under the hood)
        input_text = [str(self.transcription)]
        shap_explanations = self.text_explainer(input_text)  
        # shap_explanations typically has shape [batch, tokens, classes].
        # We'll pick the first example (index 0), then the predicted_label dimension.
        # i.e. shap_explanations[0, :, self.predicted_label].
    
        # Make sure we handle the predicted label dimension safely
        if self.predicted_label is None:
            raise ValueError("self.predicted_label is None. Inference may not have been run.")
    
        # The snippet below extracts the single (row, class) explanation:
        single_shap_values = shap_explanations[0, :, self.predicted_label]
        # 'single_shap_values' is now a shap.Explanation object representing the token-level SHAP
        # for whichever label is in self.predicted_label.
    
        # Decompose the hierarchical SHAP values into tokens/values
        values, clustering = unpack_shap_explanation_contents(single_shap_values)
        tokens, merged_values, group_sizes = process_shap_values(
            single_shap_values.data,
            values,
            grouping_threshold=grouping_threshold,
            separator=separator,
            clustering=clustering
        )
    
        # Build the dictionary: token -> SHAP value
        shap_dict = {}
        for token, val in zip(tokens, merged_values):
            # It’s good practice to cast the SHAP value to a plain float
            shap_dict[token] = float(val)
    
        return shap_dict, shap_explanations