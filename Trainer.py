import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device, config=None):
        """
        Initializes the Trainer.
        
        Parameters:
          model: The GNN model.
          train_loader: DataLoader for training data.
          test_loader: DataLoader for test data.
          optimizer: Optimizer for training.
          criterion: Loss function (e.g., BCEWithLogitsLoss with reduction='none').
          device: Device to run on (e.g., torch.device('cuda') or torch.device('cpu')).
          config (dict, optional): Additional configuration parameters.
            For example:
              - "use_weighting": bool (if True, apply feature-based sample weighting)
              - "feature_dim": int (dimension of the computed pair feature part)
              - "motif_dim": int (dimension of the motif label part)
              - "importance": tuple (importance_pair, importance_motif)
              - "num_epochs": int (number of training epochs)
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config if config is not None else {}

    def train_one_epoch(self):
        """
        Trains the model for one epoch.
        
        If 'use_weighting' is set in the configuration, applies sample weighting based on
        the active portions of the target vector.
        """
        self.model.train()
        total_loss = 0.0
        use_weighting = self.config.get("use_weighting", False)
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(data.x, data.edge_index, data.batch)  # [batch_size, out_dim]
            batch_size, out_dim = logits.size()
            target = data.y.float().view(batch_size, out_dim)
            
            # Compute per-element loss.
            loss = self.criterion(logits, target)  # shape: [batch_size, out_dim]
            
            if use_weighting:
                # Expect configuration keys "feature_dim", "motif_dim", and "importance" (a tuple).
                feature_dim = self.config.get("feature_dim")
                motif_dim = self.config.get("motif_dim")
                importance = self.config.get("importance", (1.0, 1.0))
                
                # Compute sample-wise loss and determine which graphs have active pair/motif features.
                sample_loss = loss.mean(dim=1)  # shape: [batch_size]
                pair_mask = (target[:, :feature_dim].sum(dim=1) > 0)
                motif_mask = (target[:, feature_dim:].sum(dim=1) > 0)
                w = torch.ones(batch_size, device=target.device)
                w = torch.where(pair_mask & ~motif_mask,
                                torch.full((batch_size,), importance[0], device=target.device),
                                w)
                w = torch.where(motif_mask & ~pair_mask,
                                torch.full((batch_size,), importance[1], device=target.device),
                                w)
                w = torch.where(pair_mask & motif_mask,
                                torch.full((batch_size,), max(importance[0], importance[1]), device=target.device),
                                w)
                w_expanded = w.unsqueeze(1).expand_as(loss)
                loss = loss * w_expanded  # apply weighting
            
            mean_loss = loss.mean()
            mean_loss.backward()
            self.optimizer.step()
            total_loss += mean_loss.item()
        
        return total_loss / len(self.train_loader)

    def train(self, num_epochs):
        """
        Trains the model for the given number of epochs.
        """
        for epoch in range(1, num_epochs + 1):
            epoch_loss = self.train_one_epoch()
            print(f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f}")

    @staticmethod
    def is_pure_graph(target_vec):
        """
        Determines whether a graph is "pure" by checking if its target vector is one-hot.
        A one-hot vector has exactly one element equal to 1.
        """
        return target_vec.sum().item() == 1.0

    def evaluate(self):
        """
        Evaluates the model on the test set.
        
        Computes overall loss and exact-match accuracy. Additionally, it collects predictions
        and hidden representations for graphs whose target vector is one-hot ("pure" graphs) and
        prints average predictions and hidden embeddings per target type.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        preds_dict = {}       # key: target tuple -> list of prediction tensors
        graph_repr_dict = {}  # key: target tuple -> list of graph representations
        
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                logits = self.model(data.x, data.edge_index, data.batch)  # [batch_size, out_dim]
                batch_size, out_dim = logits.size()
                target = data.y.float().view(batch_size, out_dim)
                loss = self.criterion(logits, target)
                batch_loss = loss.mean().item() * batch_size
                total_loss += batch_loss
                total_samples += batch_size
                
                # Compute predictions via sigmoid thresholding at 0.5.
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                correct = (pred == target).float().mean(dim=1)
                total_correct += correct.sum().item()
                
                # Collect predictions and representations for pure graphs.
                for i in range(batch_size):
                    t_vec = target[i].cpu()
                    if not self.is_pure_graph(t_vec):
                        continue
                    key = tuple(int(round(x)) for x in t_vec.tolist())
                    if key not in preds_dict:
                        preds_dict[key] = []
                        graph_repr_dict[key] = []
                    preds_dict[key].append(pred[i].cpu())
                    # Assume the model provides a get_graph_repr() method.
                    graph_repr = self.model.get_graph_repr(data.x, data.edge_index, data.batch)[i].cpu()
                    graph_repr_dict[key].append(graph_repr)
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        print(f"Overall Loss: {avg_loss:.4f}, Accuracy (exact match): {avg_accuracy * 100:.2f}%")
        
        print("\n=== Average Predictions (Pure Graphs) ===")
        avg_predictions = {}
        for k, preds in preds_dict.items():
            preds_tensor = torch.stack(preds)
            avg_pred = preds_tensor.float().mean(dim=0)
            avg_predictions[k] = avg_pred
            print(f"Target {k} => Average Prediction: {avg_pred.tolist()}")
        
        print("\n=== Average Hidden Embeddings (Pure Graphs) ===")
        avg_embeddings = {}
        for k, reps in graph_repr_dict.items():
            avg_repr = torch.stack(reps).mean(dim=0)
            avg_embeddings[k] = avg_repr
            print(f"Target {k} => Average Hidden Embedding: {avg_repr.tolist()}")
        
        return avg_loss, avg_accuracy, preds_dict, avg_embeddings, avg_predictions

