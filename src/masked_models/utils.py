import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM


def model_init(handle, generative=False):
    """
    Initialize the model and tokenizer based on the `handle`
    """
    if generative:
        model, tokenizer = AutoModelForCausalLM.from_pretrained(handle), AutoTokenizer.from_pretrained(handle)
    else:
        model, tokenizer = AutoModelForMaskedLM.from_pretrained(handle), AutoTokenizer.from_pretrained(handle)
    if torch.cuda.is_available():
        model = model.to('cuda:0')
    return model, tokenizer


def masked_logprob_score(sen1, sen2, tokenizer, model, device, diagnose=False):
    """
    Calculate `mask_logprob` for `sen1`. The tokens that are different compared to `sen2` are masked.
    """
    original_tokens = tokenize(sen1, tokenizer).to(device)
    masked_tokens = tokenize_with_mask(sen1, sen2, tokenizer).to(device)
    probs = model(**masked_tokens).logits.softmax(dim=-1)
    probs_true = torch.gather( 
        probs[0],
        dim=1,
        index=torch.t(original_tokens['input_ids'])
    )
    mask_indices = masked_tokens['input_ids'][0] == tokenizer.mask_token_id
    logprob = torch.mean(torch.log10(probs_true[mask_indices])).item()
    
    if diagnose:
        print('Original sentence:', sen1)
        print('Token ids:', original_tokens['input_ids'][0])
        print('Token ids (masked):', masked_tokens['input_ids'][0])
        print('Tokens:', ', '.join('`' + tokenizer.decode([t]) + '`' for t in original_tokens['input_ids'][0]))
        print('Decoded token ids:', tokenizer.decode(original_tokens['input_ids'][0]))
        print('Decoded token ids (masked):', tokenizer.decode(masked_tokens['input_ids'][0]))
        print('Probs:', probs)
        print('Probs for correct tokens:', probs_true)
        print('Probs for masked tokens:', probs_true[mask_indices])
        print('Log of their mean:', logprob)
    return logprob


def token_id(token, model_handle):
    """
    This is 'overfitted' to handle the models in our selection. It handles several edge cases:
    
    1. Some models (e.g., gpt2) have different tokens for the same tokens if they are at the start of the sentence
    and if they are not. This is an expected behavior and they were trained this way. To simulate this, we use `add_prefix_space=True`

    ```
    gpt2_tokenizer.encode('he he')
    >>> [258, 339]
    ```

    2. Other models (e.g., mistral) handle the same problems with special start tokens. To handle this we return the last token.

    ```
    mistral_tokenizer.encode('he he')
    >>> [1, 400, 400]
    ```
    """
    tokenizer = AutoTokenizer.from_pretrained(model_handle, add_prefix_space=True)
    token_ids = tokenizer.encode(token)
    return token_ids[-1]


def generative_score(sample, template, male_token_id, female_token_id, model, tokenizer, device):
    text = template(sample)
    tokens = tokenizer(text, return_tensors='pt').to(device)
    logits = model(**tokens).logits[0][-1]
    probs = logits.softmax(dim=-1)
    male_prob = probs[male_token_id]
    female_prob = probs[female_token_id]
    return float(torch.log10(male_prob) - torch.log10(female_prob))


def tokenize(sen, tokenizer, only_ids=False, **kwargs):
    """
    Use `tokenizer` to parse sentence `sen`.
    
    `only_ids` - Return only token ids if True, `BatchEncoding` otherwise.
    `kwargs` - Redirected to `tokenizer`.
    """
    batch_encoding = tokenizer(sen, return_tensors="pt", **kwargs)
    if only_ids:
        return batch_encoding['input_ids'][0].tolist()
    else:
        return batch_encoding


def tokenize_with_mask(sen1, sen2, tokenizer, only_ids=False):
    '''
    Tokenize `sen1` and replace any tokens that are different in `sen2` with a mask token.
    
    `only_ids` - Return only token ids if True, `BatchEncoding` otherwise.
    '''
    batch_encoding = tokenize(sen1, tokenizer)
    sen1_tokens, sen2_tokens = tokenize(sen1, tokenizer, only_ids=True), tokenize(sen2, tokenizer, only_ids=True)
    
    for token_id, (sen1_token, sen2_token) in enumerate(zip(sen1_tokens, sen2_tokens)):
        if sen1_token != sen2_token:
            batch_encoding['input_ids'][0][token_id] = tokenizer.mask_token_id
   
    if only_ids:
        return batch_encoding['input_ids'][0].tolist()
    else:
        return batch_encoding