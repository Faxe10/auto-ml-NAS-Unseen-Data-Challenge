import time
import copy
from sklearn.metrics import accuracy_score

import torch
from torch import optim
import torch.nn as nn

from helpers import show_time


class Trainer:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    """

    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

        # Optimizer: AdamW is generally robust for NAS tasks
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        # Scheduler: Must use Plateau since total epochs are unknown due to time constraints
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)

        # For Mixed Precision (Speedup)
        self.scaler = torch.amp.GradScaler('cuda')

        # Read the time budget from NAS (default to 1 hour if missing)
        # self.allocated_training_time = self.metadata.get('target_training_time_seconds', 3600)

    """
    ====================================================================================================================
    TRAIN ==============================================================================================================
    ====================================================================================================================
    """

    def train(self):
        self.model = self.model.to(self.device)
        t_start = time.time()
        epoch = 0

        # Early Stopping & Checkpointing Setup
        best_valid_acc = -1.0
        best_model_state = copy.deepcopy(self.model.state_dict())
        patience = 50  # Stop if no improvement after 5 epochs
        epochs_without_improvement = 0

        # Safety buffers for the clock
        PREDICTION_BUFFER_SECONDS = 45
        self.avg_epoch_time = 0.0  # Initial guess to start the loop safely

        while True:
            elapsed_time = time.time() - t_start
            # time_left_in_budget = self.allocated_training_time - elapsed_time
            global_time_left = self.clock.check()

            # --- THE DUAL CLOCK FAILSAFE ---
            # Stop if we exceed our training budget OR if the global competition clock is dying
            if epoch > 0 and global_time_left < (self.avg_epoch_time + PREDICTION_BUFFER_SECONDS):
                print(f"\n⏰ Global time budget reached! Halting cleanly after {epoch} epochs to ensure submission.")
                break

            epoch_start = time.time()
            self.model.train()
            labels, predictions = [], []

            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass with Mixed Precision
                with torch.amp.autocast('cuda'):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                # Backward pass with Scaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Store labels and predictions to compute accuracy
                labels.extend(target.cpu().tolist())
                predictions.extend(torch.argmax(output, 1).detach().cpu().tolist())

            train_acc = accuracy_score(labels, predictions)
            valid_acc = self.evaluate()

            # Step the scheduler based on validation accuracy
            self.scheduler.step(valid_acc)

            # Checkpoint the best model & Early Stopping Logic
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0  # Reset patience counter
            else:
                epochs_without_improvement += 1

            # Update average epoch time for the clock check
            epoch_time = time.time() - epoch_start
            self.avg_epoch_time = (time.time() - t_start) / (epoch + 1)

            print(
                f"\tEpoch {epoch + 1:>3} | Train Acc: {train_acc * 100:>6.2f}% | Valid Acc: {valid_acc * 100:>6.2f}% | T/Epoch: {show_time(epoch_time):<7} |")

            # Early Stopping Trigger
            if epochs_without_improvement >= patience:
                print(
                    f"\n🛑 Early stopping triggered! Validation accuracy hasn't improved in {patience} epochs. Preventing overfitting.")
                break

            epoch += 1

        print(f"  Total training runtime: {show_time(time.time() - t_start)}")

        # Restore the best model weights
        if best_valid_acc > -1.0:
            print(f"  Restoring best model weights (Val Acc: {best_valid_acc * 100:.2f}%)")
            self.model.load_state_dict(best_model_state)

        return self.model

    """
    ====================================================================================================================
    EVALUATE ===========================================================================================================
    ====================================================================================================================
    """

    def evaluate(self):
        self.model.eval()
        labels, predictions = [], []

        # CRITICAL: Disable gradients for evaluation to save VRAM and speed up
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                data = data.to(self.device)
                output = self.model(data)
                labels.extend(target.cpu().tolist())
                predictions.extend(torch.argmax(output, 1).detach().cpu().tolist())

        return accuracy_score(labels, predictions)

    """
    ====================================================================================================================
    PREDICT ============================================================================================================
    ====================================================================================================================
    """

    def predict(self, test_loader):
        self.model.eval()
        predictions = []

        # CRITICAL: Disable gradients for testing
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                predictions.extend(torch.argmax(output, 1).detach().cpu().tolist())

        return predictions
