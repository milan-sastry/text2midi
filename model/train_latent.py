import os 
#print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
import torch
#print("CUDA device count:", torch.cuda.device_count())
#print("CUDA current device:", torch.cuda.current_device())
#print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
#os.environ['CUDA_VISIBLE_DEVICES']="2,3"
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import pickle
import os
import random
from tqdm import tqdm
import torch
from torch import Tensor, argmax
from evaluate import load as load_metric
import sys
import argparse
import jsonlines
from data_loader_remi import Text2MusicDataset
from transformer_model import Transformer, MultiHeadLatentAttention
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

config_file = "../configs/config.yaml"
# Load config file
with open(config_file, 'r') as f: ##args.config
    configs = yaml.safe_load(f)

batch_size = configs['training']['text2midi_model']['batch_size']
learning_rate = configs['training']['text2midi_model']['learning_rate']
epochs = configs['training']['text2midi_model']['epochs']

# Artifact folder
artifact_folder = configs['artifact_folder']
# Load encoder tokenizer json file dictionary
tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)

# Get the vocab size
vocab_size = len(tokenizer)
print("Vocab size: ", vocab_size)

caption_dataset_path = configs['raw_data']['caption_dataset_path']
# Load the caption dataset
with jsonlines.open(caption_dataset_path) as reader:
    captions = list(reader)
    captions = captions[:len(captions)//100]

def collate_fn(batch):
    """
    Collate function for the DataLoader
    :param batch: The batch
    :return: The collated batch
    """
    input_ids = [item[0].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)    
    attention_mask = [item[1].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = [item[2].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return input_ids, attention_mask, labels


# Load the dataset
dataset = Text2MusicDataset(configs, captions, remi_tokenizer=tokenizer, mode="train", shuffle = True)
data_length = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
repo_id = "amaai-lab/text2midi"

# Create the encoder-decoder model
# Initialize the model
d_model = configs['model']['text2midi_model']['decoder_d_model']  # Model dimension (same as FLAN-T5 encoder output dimension)
nhead = configs['model']['text2midi_model']['decoder_num_heads']     # Number of heads in the multiheadattention models
num_layers = configs['model']['text2midi_model']['decoder_num_layers']  # Number of decoder layers
max_len = configs['model']['text2midi_model']['decoder_max_sequence_length']  # Maximum length of the input sequence
use_moe = configs['model']['text2midi_model']['use_moe'] # Use mixture of experts
num_experts = configs['model']['text2midi_model']['num_experts'] # Number of experts in the mixture of experts
dim_feedforward = configs['model']['text2midi_model']['decoder_intermediate_size'] # Dimension of the feedforward network model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

v_head_dim = 96
nope_head_dim = 48
ope_head_dim = 48
q_lora_rank = 64
kv_lora_rank = 64

latent_dimension = (v_head_dim, nope_head_dim, ope_head_dim, q_lora_rank, kv_lora_rank)

# model = Transformer(vocab_size, d_model, nhead, max_len, num_layers, dim_feedforward, latent_dimension, use_moe, num_experts, device=device)
model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
model = Transformer(vocab_size, 
                    d_model=768, 
                    nhead=8, 
                    max_len=2048, 
                    num_decoder_layers=18, 
                    dim_feedforward=1024, 
                    latent_dimensions=latent_dimension, 
                    use_moe=False, 
                    num_experts=8, 
                    device=device)
state_dict = torch.load(model_path, map_location=device)
result = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(result.missing_keys)}")
print(f"Unexpected keys: {len(result.unexpected_keys)}")
if len(result.missing_keys) > 0:
    print(f"First few missing keys: {result.missing_keys[:5]}")
if len(result.unexpected_keys) > 0:
    print(f"First few unexpected keys: {result.unexpected_keys[:5]}")

#print(pretrained_model.decoder.layers[9].self_attn)



# for layer in range(15, 18):
#     self_attn = model.decoder.layers[layer].self_attn
#     # num_params = sum(p.numel() for p in self_attn.parameters())
#     # print(f"Number of parameters: {num_params}")
#     latent_attn = MultiHeadLatentAttention(
#         embed_dim=self_attn.embed_dim,
#         num_heads=self_attn.heads,
#         latent_dimensions=latent_dimension,
#         dropout=0.1,
#         batch_first=self_attn.batch_first)
    
#     model.decoder.layers[layer].self_attn = latent_attn

print(model)


def print_attention_weights_comparison(model):
    print("\n===== SELF-ATTENTION WEIGHTS (First few layers) =====")
    
    # Print weights for first 3 layers with regular self-attention
    for layer_idx in range(0, 3):
        attn = model.decoder.layers[layer_idx].self_attn
        print(f"\nLayer {layer_idx} Self-Attention:")
        
        # Print named parameters with shape and stats
        for name, param in attn.named_parameters():
            print(f"  {name}: shape={param.shape}")
            print(f"    mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            # Print a small sample of values (first 5)
            print(f"    sample values: {param.data.flatten()[:5].tolist()}")
    
    print("\n===== LATENT ATTENTION WEIGHTS (Last 3 layers) =====")
    
    # Print weights for last 3 layers with latent attention
    for layer_idx in range(15, 18):
        latent_attn = model.decoder.layers[layer_idx].self_attn
        print(f"\nLayer {layer_idx} Latent Attention:")
        
        # Print named parameters with shape and stats
        for name, param in latent_attn.named_parameters():
            print(f"  {name}: shape={param.shape}")
            print(f"    mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            # Print a small sample of values (first 5)
            print(f"    sample values: {param.data.flatten()[:5].tolist()}")
        
# print_attention_weights_comparison(model)

for param in model.parameters():
    param.requires_grad = False

for layer in range(15, 18):
    self_attn = model.decoder.layers[layer].self_attn
    assert isinstance(self_attn, MultiHeadLatentAttention), f"Layer {layer} is not a MultiHeadLatentAttention layer"
    for param in self_attn.parameters():
        param.requires_grad = True

for param in model.projection.parameters():
    param.requires_grad = True

# Print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")
# Print number of trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")

print_every = 10
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
torch.cuda.empty_cache()
def train_latent(model, dataloader, criterion, num_epochs, optimizer=None, data_length=1000):   
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(total=int(data_length/batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for step, batch in enumerate(dataloader):
                print(f"Step {step}/{len(dataloader)}")
                optimizer.zero_grad()
                
                # Get the batch
                encoder_input, attention_mask, tgt = batch
                # print(encoder_input.shape)
                encoder_input = encoder_input.to(device)
                attention_mask = attention_mask.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                if use_moe:
                    outputs, aux_loss = model(encoder_input, attention_mask, tgt_input)
                else:
                    outputs = model(encoder_input, attention_mask, tgt_input)
                    aux_loss = 0

                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
                loss += aux_loss
            
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                if step % print_every == 0:
                    pbar.set_postfix({"Loss": loss.item()})
                    pbar.update(1)
            
            pbar.set_postfix({"Loss": total_loss / len(dataloader)})
            pbar.update(1)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

print(model)
# Train the model

train_latent(model, dataloader, criterion, num_epochs=epochs, optimizer=optimizer, data_length=data_length)

# Save the trained model
#torch.save(model.state_dict(), "transformer_decoder_remi_plus.pth")
#print("Model saved as transformer_decoder_remi_plus.pth")
