import torch

def model_predict_lime(texts,model,tokenizer,device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    return outputs.detach().cpu().numpy()
