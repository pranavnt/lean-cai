import json
from typing import List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Subset
import wandb

class RewardHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        return self.reward_head(hidden_states)

class ComparisonDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train(model, tokenizer, train_dataset, val_dataset, epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        for batch in train_dataloader:
            prompt = batch["prompt"][0]
            chosen = batch["chosen"][0]
            rejected = batch["rejected"][0]

            chosen_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ]

            rejected_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected},
            ]

            chosen_inputs = tokenizer.apply_chat_template(chosen_messages, padding=True, truncation=True, return_tensors="pt").to(device)
            rejected_inputs = tokenizer.apply_chat_template(rejected_messages, padding=True, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                chosen_outputs = model(chosen_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
                rejected_outputs = model(rejected_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]

            chosen_rewards = model.reward_head(chosen_outputs)
            rejected_rewards = model.reward_head(rejected_outputs)

            reward_diff = chosen_rewards - rejected_rewards
            target = torch.ones_like(reward_diff)

            loss = loss_fn(reward_diff, target)
            total_loss += loss.item()

            accuracy = ((reward_diff > 0).float() == target).float().mean()
            total_accuracy += accuracy.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"batch_train_loss": loss.item(), "batch_train_accuracy": accuracy.item()}, step=epoch)

        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_accuracy / len(train_dataloader)
        wandb.log({"train_loss": avg_loss, "train_accuracy": avg_accuracy}, step=epoch)

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_dataloader:
                prompt = batch["prompt"][0]
                chosen = batch["chosen"][0]
                rejected = batch["rejected"][0]

                chosen_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ]

                rejected_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected},
                ]

                chosen_inputs = tokenizer.apply_chat_template(chosen_messages, padding=True, truncation=True, return_tensors="pt").to(device)
                rejected_inputs = tokenizer.apply_chat_template(rejected_messages, padding=True, truncation=True, return_tensors="pt").to(device)

                chosen_outputs = model(**chosen_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
                rejected_outputs = model(**rejected_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]

                chosen_rewards = model.reward_head(chosen_outputs)
                rejected_rewards = model.reward_head(rejected_outputs)

                reward_diff = chosen_rewards - rejected_rewards
                target = torch.ones_like(reward_diff)

                loss = loss_fn(reward_diff, target)
                val_loss += loss.item()

                accuracy = ((reward_diff > 0).float() == target).float().mean()
                val_accuracy += accuracy.item()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_accuracy = val_accuracy / len(val_dataloader)
        wandb.log({"val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy}, step=epoch)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Train Accuracy: {avg_accuracy:.4f} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {avg_val_accuracy:.4f}")

    return model

# Initialize wandb
wandb.init(project="llama-reward-modeling")

# Load the pre-trained Llama model and tokenizer
model_name = "./llama3-mathlib-sft"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Replace the model's head with the reward head
hidden_size = model.config.hidden_size
model.reward_head = RewardHead(hidden_size)

# Load the comparison dataset from JSONL file
jsonl_file = "./jsonl_final.jsonl"
dataset = ComparisonDataset(jsonl_file)

# Split the dataset into train and validation sets
val_size = 300
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Train the model
epochs = 5
batch_size = 1  # Increase the batch size
learning_rate = 1e-5
trained_model = train(model, tokenizer, train_dataset, val_dataset, epochs, batch_size, learning_rate)