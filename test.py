import pickle
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from model.transformer_model import Transformer
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(model, input_text):
    """Visualize attention after generation"""
    # Get attention weights from the last decoder layer's self attention
    # This assumes you've already run model.generate()
    attention_weights = model.decoder.layers[-1].self_attn.attention_weights
    
    if attention_weights is None:
        print("No attention weights stored. Run generation first.")
        return
    
    # Get input tokens
    input_tokens = tokenizer.tokenize(input_text)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # We'll use the first head's attention weights
    # Shape is typically [batch, heads, seq_len, seq_len]
    attn = attention_weights[0, 0, :, :].numpy()
    
    # Plot heatmap (limit to a reasonable size if too large)
    max_tokens = min(30, attn.shape[0])  # Limit to 30 tokens for readability
    
    # Create readable token labels
    token_labels = input_tokens[:max_tokens]
    
    sns.heatmap(attn[:max_tokens, :max_tokens], 
                xticklabels=token_labels,
                yticklabels=token_labels,
                cmap='viridis')
    
    plt.title('Attention Map - Last Layer, First Head')
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=300)
    plt.show()
    
    return attn

repo_id = "amaai-lab/text2midi"
# Download the model.bin file
model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
# Download the vocab_remi.pkl file
tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")

# Load the tokenizer dictionary
with open(tokenizer_path, "rb") as f:
    r_tokenizer = pickle.load(f)

# Get the vocab size
vocab_size = len(r_tokenizer)
print("Vocab size: ", vocab_size)

v_head_dim = 96
nope_head_dim = 48
ope_head_dim = 48
q_lora_rank = 64
kv_lora_rank = 64

latent_dimensions = (v_head_dim, nope_head_dim, ope_head_dim, q_lora_rank, kv_lora_rank)
# self attention
#model = Transformer(False,vocab_size, 768, 8, 2048, 18, 1024, latent_dimensions, False, 8, device=device)

# latent attention
model = Transformer(True,vocab_size, 768, 8, 2048, 18, 1024, latent_dimensions, False, 8, device=device)

print(model)
# when loading just self attention, initialize Transformer with latent_dimensions set to None, pass False as first parameter
#model.load_state_dict(torch.load(model_path, map_location=device))
latent_model_path = './last_block_latent.pth'
self_model_path = './last_block_self.pth'
#strict = false for self attention, because of freq_cos and freq_sin
model.load_state_dict(torch.load(latent_model_path, map_location=torch.device('cpu')),strict=True)
model.eval()
# confirm Transformer architecture corresponds to the dict you are loading
print(model)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

print('Model loaded.')


# Enter the text prompt and tokenize it
src = "A melodic classical piano song  Set in G minor with a 4/4 time signature, it moves at a lively Presto tempo. The composition evokes a blend of relaxation and darkness, with hints of happiness and a meditative quality."
print('Generating for prompt: ' + src)

inputs = tokenizer(src, return_tensors='pt', padding=True, truncation=True)
input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
input_ids = input_ids.to(device)
attention_mask =nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0) 
attention_mask = attention_mask.to(device)

# Generate the midi
output = model.generate(input_ids, attention_mask, max_len=2048,temperature = 1.0)
output_list = output[0].tolist()
generated_midi = r_tokenizer.decode(output_list)
generated_midi.dump_midi("output.mid")

#attention_matrix = visualize_attention(model, src)
#print("Attention visualization saved to 'attention_visualization.png'")
