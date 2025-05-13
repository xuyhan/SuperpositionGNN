import torch
import torch.optim as optim
from torch_geometric.data import DataLoader

from GraphGeneration import SyntheticGraphDataGenerator
from Model            import GNNModel
from Trainer          import Trainer          # includes geometry_analysis, etc.
from Writer           import get_writer
from datasets.tox21_loader import get_tox21_loaders   # real-data loader


# ───────────────────────── helper: tiny wrapper for TensorBoard graphs ──────────
class ModelWrapper(torch.nn.Module):
    def __init__(self, gnn_model):
        super().__init__()
        self.gnn_model = gnn_model

    def forward(self, x, edge_index, batch):
        return self.gnn_model(x, edge_index, batch)


# ────────────────────────────────────────────────────────────────────────────────
def run_multiple_experiments(experiment_config, num_experiments=10):
    results, all_model_params, all_average_embeddings, empty_graph_stats_list = [], [], [], []

    for i in range(num_experiments):
        print(f"\nRunning experiment {i + 1}/{num_experiments}…")
        experiment_number = i + 1

        # ── 1.  Data: real Tox21 or synthetic ───────────────────────────────────
        if experiment_config["mode"].lower() == "tox21":
            train_loader, test_loader, in_dim_auto, pos_weight = get_tox21_loaders(
                root         = experiment_config.get("root", "data/Tox21"),
                batch_size   = experiment_config.get("batch_size", 32),
                train_split  = experiment_config.get("train_split", 0.8),
                shuffle      = True,
                mask_missing = experiment_config.get("mask_missing", True)
            )
            print("Per-task pos_weight:", pos_weight)

            in_dim     = in_dim_auto
            out_dim    = 12          # fixed for Tox21
            motif_dim  = 0
            hidden_dims = experiment_config.get("hidden_dims", [64, 64])

        else:   # synthetic / motif / count …
            generator = SyntheticGraphDataGenerator(
                mode          = experiment_config.get("mode", "simple"),
                num_categories= experiment_config.get("num_categories", 3),
                p             = experiment_config.get("p", 0.25),
                p_count       = experiment_config.get("p_count", 0.9),
                num_nodes     = experiment_config.get("num_nodes", 20),
                motif_dim     = experiment_config.get("motif_dim", 0),
                chain_length_min  = experiment_config.get("chain_length_min", 2),
                chain_length_max  = experiment_config.get("chain_length_max", 7),
                candidate_matrices= experiment_config.get("candidate_matrices", None)
            )
            train_data = generator.generate_data(num_samples=experiment_config.get("num_train_samples", 5000))
            test_data  = generator.generate_data(num_samples=experiment_config.get("num_test_samples", 2000))

            train_loader = DataLoader(train_data, batch_size=experiment_config.get("batch_size", 4), shuffle=True)
            test_loader  = DataLoader(test_data , batch_size=experiment_config.get("batch_size", 4), shuffle=False)

            in_dim      = experiment_config.get("in_dim", 3)
            hidden_dims = experiment_config.get("hidden_dims", [4, 4])
            motif_dim   = experiment_config.get("motif_dim", 0)
            out_dim     = experiment_config.get("num_categories", 3) + motif_dim

        # ── 2.  Build model ─────────────────────────────────────────────────────
        pooling = experiment_config.get("pooling", "mean")

        model = GNNModel(
            model_type   = experiment_config.get("model_type", "GCN"),
            in_dim       = in_dim,
            hidden_dims  = hidden_dims,
            out_dim      = out_dim,
            freeze_final = True,
            pooling      = pooling,
            gm_p         = experiment_config.get("gm_p", 1.0)
        ).to(experiment_config["device"])

        # ── 3.  Criterion (with / without pos_weight) ───────────────────────────
        if experiment_config.get("loss", "BCE") == "BCE":
            if experiment_config["mode"].lower() == "tox21":
                criterion = torch.nn.BCEWithLogitsLoss(
                    reduction='none',
                    pos_weight=pos_weight.to(experiment_config["device"])
                )
            else:
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif experiment_config.get("loss", "BCE") == "MSE":
            criterion = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError("Unsupported loss type")

        optimizer = optim.Adam(model.parameters(), lr=experiment_config.get("lr", 0.01))

        # ── 4.  TensorBoard setup (optional) ────────────────────────────────────
        writer = get_writer(experiment_config.get("log_dir", None))
        if experiment_config.get("add_graph", False):
            wrapped = ModelWrapper(model)
            batch_sample = next(iter(train_loader))
            writer.add_graph(wrapped, (batch_sample.x, batch_sample.edge_index, batch_sample.batch))

        # ── 5.  Trainer configuration dict ─────────────────────────────────────
        trainer_cfg = {
            "use_weighting"      : experiment_config.get("use_weighting", False),
            "feature_dim"        : experiment_config.get("num_categories", 3),
            "motif_dim"          : motif_dim,
            "importance"         : experiment_config.get("importance", (15.0, 10.0)),
            "num_epochs"         : experiment_config.get("num_epochs", 5),
            "phase1_epochs"      : experiment_config.get("phase1_epochs", 0 if experiment_config["mode"].lower()=="tox21" else 5),
            "phase2_epochs"      : experiment_config.get("phase2_epochs", 10),
            "log_dir"            : experiment_config.get("log_dir", None),
            "track_embeddings"   : experiment_config.get("track_embeddings", False),
            "track_singular_values": experiment_config.get("track_singular_values", False),
            "mode"               : experiment_config["mode"],
            "per_label_thresholds": experiment_config["mode"].lower() == "tox21",
        }

        # ── 6.  Train phases ────────────────────────────────────────────────────
        trainer = Trainer(model, train_loader, test_loader, optimizer, criterion,
                          experiment_config["device"], trainer_cfg)

        # Phase-1 (may be 0 epochs)
        trainer.train(num_epochs=trainer_cfg["phase1_epochs"], experiment_number=experiment_number)

        # Un-freeze final layer for phase-2
        model.lin_out.weight.requires_grad = True
        if model.lin_out.bias is not None:
            model.lin_out.bias.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=experiment_config.get("lr", 0.01))
        trainer.optimizer = optimizer
        trainer.train(num_epochs=trainer_cfg["phase2_epochs"], experiment_number=experiment_number)

        # ── 7.  Fit thresholds once at the very end (Tox21 only) ───────────────
        if trainer_cfg["per_label_thresholds"]:
            print("Fitting per-label thresholds on validation set …")
            trainer._fit_thresholds(test_loader)          # use a val_loader if you have one


        # ── 8.  Evaluation & bookkeeping ───────────────────────────────────────
        model_params = extract_model_parameters(model)
        all_model_params.append(model_params)

        avg_loss, _, _, avg_embeddings, avg_predictions, empty_stats = trainer.evaluate()
        res = trainer.structure_of_representation(out_dim, avg_predictions, avg_embeddings, avg_loss)

        results.append(res)
        all_average_embeddings.append(avg_embeddings)
        empty_graph_stats_list.append(empty_stats)

    return results, all_model_params, all_average_embeddings, empty_graph_stats_list, avg_predictions


# ───────────────────────────────── helper ───────────────────────────────────────
def extract_model_parameters(model):
    return {name: p.detach().cpu().tolist() for name, p in model.named_parameters()}