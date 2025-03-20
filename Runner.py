import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from GraphGeneration import SyntheticGraphDataGenerator
from Model import *
from Trainer import Trainer  # Trainer now includes geometry_analysis, etc.
from Writer import get_writer

# Define a wrapper for your GNN model.
class ModelWrapper(torch.nn.Module):
    def __init__(self, gnn_model):
        super(ModelWrapper, self).__init__()
        self.gnn_model = gnn_model

    def forward(self, x, edge_index, batch):
        return self.gnn_model(x, edge_index, batch)

def run_multiple_experiments(experiment_config, num_experiments=10):
    results = []
    all_model_params = []
    all_average_embeddings = []
    for i in range(num_experiments):
        print(f"\nRunning experiment {i+1}/{num_experiments}...")
        experiment_number = i + 1
        
        # Create data generator based on experiment type (mode)
        generator = SyntheticGraphDataGenerator(
            mode=experiment_config.get("mode", "simple"),
            num_categories=experiment_config.get("num_categories", 3),
            p=experiment_config.get("p", 0.25),
            p_count=experiment_config.get("p_count", 0.9),  # Probability for edges in count mode
            num_nodes=experiment_config.get("num_nodes", 20),
            motif_dim=experiment_config.get("motif_dim", 0),  # 0 for simple mode
            chain_length_min=experiment_config.get("chain_length_min", 2),
            chain_length_max=experiment_config.get("chain_length_max", 7),
            candidate_matrices=experiment_config.get("candidate_matrices", None)
        )
        train_data = generator.generate_data(num_samples=experiment_config.get("num_train_samples", 5000))
        test_data = generator.generate_data(num_samples=experiment_config.get("num_test_samples", 2000))
        
        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=experiment_config.get("batch_size", 4), shuffle=True)
        test_loader = DataLoader(test_data, batch_size=experiment_config.get("batch_size", 4), shuffle=False)
        
        # Instantiate model using your actual GNNModel.
        in_dim = experiment_config.get("in_dim", 3)
        hidden_dims = experiment_config.get("hidden_dims", [4, 4])
        motif_dim = experiment_config.get("motif_dim", 0)
        # For a "simple" experiment, output dimension = num_categories + motif_dim (here motif_dim is 0)
        out_dim = experiment_config.get("num_categories", 3) + motif_dim
        pooling = experiment_config.get("pooling", "mean")
        
        model = GNNModel(
            model_type=experiment_config.get("model_type", "GCN"),
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            freeze_final=True,
            pooling=pooling
        ).to(experiment_config["device"])
        
        optimizer = optim.Adam(model.parameters(), lr=experiment_config.get("lr", 0.01))
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        # Save the model as a graph to visualize in TensorBoard.
        writer = get_writer(experiment_config.get("log_dir", None))
        # Create a wrapper around your model so that TensorBoard can trace its forward pass.
        wrapped_model = ModelWrapper(model)
        batch_data = next(iter(train_loader))
        if experiment_config.get("add_graph", False) == True:
            writer.add_graph(wrapped_model, (batch_data.x, batch_data.edge_index, batch_data.batch))
        
        # Trainer configuration
        trainer_config = {
            "use_weighting": experiment_config.get("use_weighting", False),
            "feature_dim": experiment_config.get("num_categories", 3),
            "motif_dim": motif_dim,
            "importance": experiment_config.get("importance", (15.0, 10.0)),
            "num_epochs": experiment_config.get("num_epochs", 5),
            "phase1_epochs": experiment_config.get("phase1_epochs", 5),
            "phase2_epochs": experiment_config.get("phase2_epochs", 10),
            "log_dir": experiment_config.get("log_dir", None),
            "track_embeddings": experiment_config.get("track_embeddings", False),
        }
        
        # Create Trainer instance and train the model.
        trainer = Trainer(model, train_loader, test_loader, optimizer, criterion, experiment_config["device"], trainer_config)
        trainer.train(num_epochs=trainer_config["phase1_epochs"], experiment_number=experiment_number)
        
        # Phase 2: Unfreeze final layer and continue training.
        model.lin_out.weight.requires_grad = True
        if model.lin_out.bias is not None:
            model.lin_out.bias.requires_grad = True

        # Reinitialize optimizer with updated parameters.
        optimizer = optim.Adam(model.parameters(), lr=experiment_config.get("lr", 0.01))
        trainer.optimizer = optimizer  # Update the trainer's optimizer.
        trainer.train(num_epochs=trainer_config["phase2_epochs"], experiment_number=experiment_number)

        # Extract model parameters for analysis.
        model_params = extract_model_parameters(model)
        all_model_params.append(model_params)

        # Evaluate the model using the Trainer instance.
        avg_loss, __, __, avg_embeddings, avg_predictions = trainer.evaluate()
        total_target_dim = experiment_config.get("num_categories", 3) + motif_dim
        result = trainer.structure_of_representation(total_target_dim, avg_predictions, avg_embeddings, avg_loss)
        results.append(result)
        all_average_embeddings.append(avg_embeddings)
    return results, all_model_params, all_average_embeddings


def extract_model_parameters(model):
    """
    Extracts and returns a dictionary of all model parameters (weights and biases)
    from any number of layers in the given model.
    """
    params = {}
    for name, param in model.named_parameters():
        # Detach the parameter, move to CPU, and convert to a list for JSON-serialization.
        params[name] = param.detach().cpu().tolist()
    return params
