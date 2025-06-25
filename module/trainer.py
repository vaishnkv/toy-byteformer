import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch.optim import Adam


class SimpleTrainer:
    def __init__(self, model, loss_fn,dataloader, val_dataloader=None, project_name="byteformer-sanity", run_name=None):
        self.model = model.to(self._get_device())
        self.loss_fn = loss_fn
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = self._get_device()

        wandb.init(project=project_name, name=run_name)
        wandb.watch(self.model, log="all")

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs=10):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            correct,total=0.,0.0
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                
                X = batch["sample"].to(self.device)
                y = batch["target"].to(self.device)

                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                preds = y_hat.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            acc = correct / total
            print(f"Validation Accuracy: {acc:.4f}")
                

            avg_loss = total_loss / len(self.dataloader)
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
            wandb.log({"epoch": epoch + 1, "training_accuracy": acc})
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            if self.val_dataloader:
                val_acc = self.evaluate()
                wandb.log({"epoch": epoch + 1, "val_accuracy": val_acc})

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                X = batch["sample"].to(self.device)
                y = batch["target"].to(self.device)
                y_hat = self.model(X)

                preds = y_hat.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
        return acc
