import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(model_output,attention_mask):
    token_embedings = model_output[0]
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedings.size()).float()
    return torch.sum(token_embedings * input_mask_expanded,1)/torch.clamp(input_mask_expanded.sum(1),min=1e-9)

def embed_func(batch,tokenizer,model):
    # 
    if isinstance(batch,float):
        batch=""
    encoded_input = tokenizer(batch,padding = False,truncation = True,return_tensors='pt')
    
    #计算token的embedding
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output,encoded_input['attention_mask'])
    #normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings,p = 2 , dim = 1)
    return [list(sentence_embeddings[0].numpy())]
