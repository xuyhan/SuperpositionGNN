import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import math
import itertools
from Writer import get_writer

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
              Example keys:
                  "use_weighting": bool,
                  "feature_dim": int,
                  "motif_dim": int,
                  "importance": tuple,
                  "num_epochs": int
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config if config is not None else {}

        # Initialize the SummaryWriter with a common log directory.
        log_dir = self.config.get("log_dir", None)
        if log_dir:
            self.writer = get_writer(log_dir)
            print(f"TensorBoard logs will be saved to the common directory: {log_dir}")
        else:
            self.writer = None

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        use_weighting = self.config.get("use_weighting", False)
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(data.x, data.edge_index, data.batch)  # [batch_size, out_dim]
            batch_size, out_dim = logits.size()
            target = data.y.float().view(batch_size, out_dim)
            loss = self.criterion(logits, target)  # shape: [batch_size, out_dim]
            if use_weighting:
                feature_dim = self.config.get("feature_dim")
                motif_dim = self.config.get("motif_dim")
                importance = self.config.get("importance", (1.0, 1.0))
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

    def train(self, num_epochs, experiment_number=1):
        for epoch in range(1, num_epochs + 1):
            epoch_loss = self.train_one_epoch()
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
            _, avg_accuracy, _, avg_embeddings, _, _ = self.evaluate()
            if self.writer:
                # Log epoch loss for multiple experiments under the same main tag "Eval/Epoch_Loss"
                self.writer.add_scalars("Eval/Epoch_Loss", {f"exp_{experiment_number}": epoch_loss}, epoch)
                # Log accuracy for multiple experiments under the same main tag "Eval/Accuracy"
                self.writer.add_scalars("Eval/Accuracy", {f"exp_{experiment_number}": avg_accuracy}, epoch)

                if self.config.get("track_singular_values", False):
                    # Log the singular values for each target tuple.
                    # Convert the vectors to a tensor (ensuring they all have the same length):
                    embeddings = torch.stack(list(avg_embeddings.values()))
                    # Perform SVD analysis on the embeddings.
                    rank, singular_values = Trainer.svd_analysis(embeddings)
                    # Log the singular values to TensorBoard.
                    self.writer.add_scalars("Eval/Singular_Values/Rank",{f"exp_{experiment_number}": rank}, epoch)
                    scalar_dict = {f"sv_{i}": float(s) for i, s in enumerate(singular_values)}
                    self.writer.add_scalars(f"Eval/Singular_Values_exp_{experiment_number}", scalar_dict, epoch)

                if self.config.get("track_embeddings", False):
                    # Log the average embeddings for each target tuple.
                    # Convert the vectors to a tensor (ensuring they all have the same length):
                    embeddings = torch.stack(list(avg_embeddings.values()))
                    # Use the dictionary keys as metadata:
                    metadata = list(avg_embeddings.keys())
                    # Log the embeddings to TensorBoard.
                    self.writer.add_embedding(embeddings, metadata=metadata, global_step=epoch, tag=(f"Eval/Avg_Embeddings/exp_{experiment_number}"))


    @staticmethod
    def is_pure_graph(target_vec):
        # A graph is considered pure if its target vector is one-hot.
        return target_vec.sum().item() == 1.0

    @staticmethod
    def is_empty_graph(target_vec):
        # A graph is considered empty if its target vector is all zeros.
        return target_vec.sum().item() == 0.0

    @staticmethod
    def aggregate_embeddings_with_stats(embeddings_list):
        # Aggregates a list of embedding tensors by stacking them into a single tensor
        # and computing the mean and standard deviation along the first dimension.
        if not embeddings_list:
            return None, None  # or handle empty case as needed
        embeddings_tensor = torch.stack(embeddings_list)  # shape: [num_graphs, embedding_dim]
        mean = embeddings_tensor.mean(dim=0)
        std = embeddings_tensor.std(dim=0)
        return mean, std

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        preds_dict = {}         # key: target tuple -> list of prediction tensors
        graph_repr_dict = {}    # key: target tuple -> list of graph representations
        empty_graph_repr = []   # list to store embeddings for empty graphs

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

                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                correct = (pred == target).float().mean(dim=1)
                total_correct += correct.sum().item()

                for i in range(batch_size):
                    t_vec = target[i].cpu()
                    # Process pure graphs (one-hot targets)
                    if Trainer.is_pure_graph(t_vec):
                        key = tuple(int(round(x)) for x in t_vec.tolist())
                        if key not in preds_dict:
                            preds_dict[key] = []
                            graph_repr_dict[key] = []
                        preds_dict[key].append(pred[i].cpu())
                        graph_repr = self.model.get_graph_repr(data.x, data.edge_index, data.batch)[i].cpu()
                        graph_repr_dict[key].append(graph_repr)
                    # Process empty graphs (all-zero targets)
                    elif Trainer.is_empty_graph(t_vec):
                        graph_repr = self.model.get_graph_repr(data.x, data.edge_index, data.batch)[i].cpu()
                        empty_graph_repr.append(graph_repr)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        # Aggregate statistics for empty graphs if any are found.
        empty_graph_stats = None
        if empty_graph_repr:
            mean, std = Trainer.aggregate_embeddings_with_stats(empty_graph_repr)
            empty_graph_stats = {'mean': mean, 'std': std}
            # For debugging or logging, you might print or store these stats.
            # Example: print(f"Empty Graph Embedding Stats: Mean {mean.tolist()}, Std {std.tolist()}")

        # Compute average predictions for pure graphs as before.
        avg_predictions = {}
        for k, preds in preds_dict.items():
            preds_tensor = torch.stack(preds)
            avg_pred = preds_tensor.float().mean(dim=0)
            avg_predictions[k] = avg_pred

        avg_embeddings = {}
        for k, reps in graph_repr_dict.items():
            avg_repr = torch.stack(reps).mean(dim=0)
            avg_embeddings[k] = avg_repr

        return avg_loss, avg_accuracy, preds_dict, avg_embeddings, avg_predictions, empty_graph_stats

    # --- Geometric Analysis Functions ---
    @staticmethod
    def geometry_of_representation(avg_embeddings_active):
        '''
        Returns the geometry of the representation (i.e. whether or not the active embeddings 
        are in ideal configuraion).
        The code computes an ideal angle based on n vectors and m dimensions: n≤m returns 90° (orthogonality);
        n=m+1 yields a regular simplex; n>m+1 employs a heuristic, effectively reducing the angle.
        '''
        embeddings = list(avg_embeddings_active.values())
        n = len(embeddings)
        if n == 0:
            return 0, 0
        m = len(embeddings[0])
        if n <= m:
            ideal_angle = np.pi / 2
        elif n == m + 1:
            ideal_angle = np.arccos(-1.0 / m)
        else:
            ideal_angle = np.arccos(np.sqrt((n - m) / (m * (n - 1))))
        tol = 0.6  # Tolerance in radians.
        smallest_angles = []
        collapsed = 0
        for i in range(n):
            angles = []
            v = np.array(embeddings[i])
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                w = np.array(embeddings[j])
                norm_w = np.linalg.norm(w)
                if norm_w == 0:
                    continue
                cosine = np.dot(v, w) / (norm_v * norm_w)
                cosine = np.clip(cosine, -1.0, 1.0)
                angle = np.arccos(cosine)
                angles.append(angle)
            if angles:
                min_angle = min(angles)
                smallest_angles.append(min_angle)
                if min_angle < 0.1:
                    collapsed += 1
        if all(angle >= (ideal_angle - tol) for angle in smallest_angles):
            geometry = n
        else:
            geometry = 0
        return geometry, collapsed

    @staticmethod
    def active_targets_in_representation(target_dim, avg_predictions, avg_embeddings):
        '''
        Returns the number of active and accurate targets in the representation plus the 
        embeddings of the active targets to be used in the geometry analysis.
        '''
        num_active_targets = 0
        num_accurate_targets = 0
        avg_embeddings_active = {}
        sigma_accurate = 0.3
        sigma_active = 0.5
        for key, preds in avg_predictions.items():
            if all((key[i] - preds[i].item()) < sigma_active for i in range(target_dim)):
                num_active_targets += 1
                avg_embeddings_active[key] = avg_embeddings[key]
            if all(abs(key[i] - preds[i].item()) < sigma_accurate for i in range(target_dim)):
                num_accurate_targets += 1
        return num_active_targets, avg_embeddings_active, num_accurate_targets

    def structure_of_representation(self, target_dim, avg_predictions, avg_embeddings, final_loss):
        '''
        Returns the structure of the representation (i.e. the number of active targets, the number of accurate targets,
        the geometry of the representation, and whether the representation is collapsed).
        '''
        num_active_targets, avg_embeddings_active, num_accurate_targets = Trainer.active_targets_in_representation(target_dim, avg_predictions, avg_embeddings)
        geometry, collapsed = Trainer.geometry_of_representation(avg_embeddings_active)
        if geometry > 0:
            category_with_loss = [target_dim, num_active_targets, num_accurate_targets, geometry, collapsed, final_loss]
            print(f"Category_with_loss: [target_dim, num_active_targets, num_accurate_targets, geometry, collapsed, final_loss] = {category_with_loss}")
            return category_with_loss
        else:
            category_with_loss = [target_dim, num_active_targets, num_accurate_targets, "Failed", collapsed, final_loss]
            print(f"Category_with_loss: [target_dim, num_active_targets, num_accurate_targets, geometry, collapsed, final_loss] = {category_with_loss}")
            return [target_dim, num_active_targets, num_accurate_targets, "Failed", collapsed, final_loss]

    @staticmethod
    def geometry_analysis(results, all_model_params, all_average_embeddings):
        '''
        Used in pipeline after running multiple experiments.
        '''
        config_losses = {}
        model_params = {}
        average_embeddings = {}
        for i, res in enumerate(results):
            if res is None:
                continue
            key = (res[1], res[2], res[3], res[4])
            config_losses.setdefault(key, []).append(res[5])
            model_params.setdefault(key, []).append(all_model_params[i])
            average_embeddings.setdefault(key, []).append(all_average_embeddings[i])
        return config_losses, model_params, average_embeddings

    @staticmethod
    def summarize_config_losses(config_losses, model_params, average_embeddings):
        '''
        Used in pipeline after geometry analysis.
        '''
        summary = {}
        model_summary = {}
        average_embeddings_summary = {}
        for config, losses in config_losses.items():
            losses_clean = [l for l in losses if l is not None]
            avg_loss = np.mean(losses_clean)
            std_loss = np.std(losses_clean)
            summary[config] = (avg_loss, std_loss, len(losses_clean))
        for type in model_params.keys():
            model_summary[type] = model_params[type][0]
            average_embeddings_summary[type] = average_embeddings[type][0]
        return summary, model_summary, average_embeddings_summary
    
    @staticmethod
    def svd_analysis(embeddings):
        """
        Analyzes a set of average embeddings.

        Returns:
            tuple: A tuple containing:
                - rank (int): The rank of the embeddings matrix.
                - singular_values (np.ndarray): The singular values from the SVD of the matrix.
        """
        import numpy as np

        # Convert dictionary or list to a NumPy array.
        if isinstance(embeddings, dict):
            embeddings = list(embeddings.values())

            embeddings = np.array([
                embedding.cpu().detach().numpy() if hasattr(embedding, 'cpu') else embedding.numpy()
                for embedding in embeddings
            ])
    
        # Ensure embeddings is a 2D array (if it's a single vector, reshape it).
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
    
        # Compute the rank of the embeddings matrix.
        rank = np.linalg.matrix_rank(embeddings)
    
        # Perform Singular Value Decomposition (SVD)
        U, singular_values, Vt = np.linalg.svd(embeddings, full_matrices=False)
    
        return rank, singular_values