import torch

def forward_func(inputs,model, attention_mask=None):
    pred = model(inputs, input_type="token_id" , attention_mask=attention_mask)
    return pred

def forward_func_embd(input_emb,model, attention_mask=None):
    pred = model(inputs_embeds=input_emb, attention_mask=attention_mask)
    return pred.max(1).values

def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def calculate_token_pairs(row,tokenizer,ref_token_id):
    input_ids, ref_ids = construct_input_ref_pair(row['transcription'], ref_token_id)
    attention_mask = construct_attention_mask(input_ids)
    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    return input_ids, ref_ids,attention_mask,all_tokens

def construct_input_ref_pair(text, tokenizer,ref_token_id):
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    ref_input_ids =  [ref_token_id] * len(input_ids[0])
    return input_ids, torch.tensor([ref_input_ids])

def construct_whole_embeddings(input_ids,model, ref_input_ids):

    input_embeddings =model.transformer.embeddings.word_embeddings(input_ids)
    ref_input_embeddings =model.transformer.embeddings.word_embeddings(ref_input_ids)

    return input_embeddings, ref_input_embeddings

def predict_label_confidence(row,device):
    output = forward_func(row['token ids'].to(device) , row['attention_mask'].to(device))
    output = torch.softmax(output,dim=1)
    C = output[0][0].cpu().item()
    MCI = output[0][1].cpu().item()
    ADRD = output[0][2].cpu().item()
    predicted_label = torch.argmax(output).cpu().item()
    return C, MCI,ADRD,predicted_label

def model_predict_lime(texts,model,tokenizer,device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    return outputs.detach().cpu().numpy()

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions