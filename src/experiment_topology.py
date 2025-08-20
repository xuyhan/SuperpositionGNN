from typing import Dict, Any

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from src.base_model import BaseModel
from src.data import SharedDataset
from src.features import FeatureExtractor, IsInsideMotifFamily, HasMotifFamily
from src.logger import CustomLogger
from src.measure import superposition_index, wno_ambient, wno_intrinsic
from src.utils import Constants, compute_pos_weight_from_loader
from src.utils import Experiment


class ExperimentTopology(Experiment):

    def __init__(self, hidden_dim: int, pooling: str, lr: float, weight_decay: float,
                 batch_size: int, model_type: str,
                 seeds: list[int], device: torch.device, logger: CustomLogger):

        super().__init__(seeds, device, logger)

        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.model_type = model_type

    def _create_and_fit_model(self, experiment_idx: int, train_loader: DataLoader, use_cached_models: bool, weight: bool = True) -> BaseModel:

        if weight:
            pos_weight, _ = compute_pos_weight_from_loader(train_loader, 2)
            pos_weight = pos_weight.to(self.device)
        else:
            pos_weight = 0.5

        model = BaseModel(identifier=f"Exp2_GNN={self.model_type}_pool={self.pooling}_s={self.seeds[experiment_idx]}",
                          model_type=self.model_type,
                          in_dim=1,
                          hidden_dims=[self.hidden_dim, self.hidden_dim, self.hidden_dim],
                          out_dim=2,
                          num_ff=1,
                          pooling=self.pooling,
                          criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none'),
                          logger=self.logger)

        model.set_device(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.fit(train_loader, optimizer, num_epochs=1600, use_cached=use_cached_models)

        return model

    def summary(self) -> str:

        return f"hidden_dim={self.hidden_dim},pool={self.pooling},lr={self.lr},batch={self.batch_size},type={self.model_type}"

    def run_once(self, experiment_idx: int, use_cached_models: bool) -> Dict[str, Any]:

        if experiment_idx >= len(self.seeds):
            raise Exception(f"Experiment {experiment_idx} out of range")

        train_loader = DataLoader(SharedDataset.CONJUNCTION.train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(SharedDataset.CONJUNCTION.test_data, batch_size=self.batch_size, shuffle=False)

        model = self._create_and_fit_model(experiment_idx, train_loader, use_cached_models)

        _, _, avg_loss, acc_total, prec_total, rec_total, auc_total, per_class_results = model.evaluate(test_loader)

        record = {
            "loss": avg_loss,
            "acc": acc_total,
            "rec": rec_total,
            "per_class_results": per_class_results
        }

        feature_extractor = FeatureExtractor(model)

        self.logger.info(f"acc={acc_total:.3f} prec={prec_total:.3f} rec={rec_total:.3f}")

        for geometry in ["node", "graph"]:
            if geometry == "node":
                all_features, _, auc, active_mask = feature_extractor.compute_node_concept_features(
                    test_loader,
                    concept_family=IsInsideMotifFamily(cycle_lengths=(3, 4, 5, 6)),
                    layer_idx=2,
                    pre_relu=False,
                    auroc_thresh=0.6,
                    min_pos=200,
                    min_neg=200,
                )
            else:
                all_features, _, auc, active_mask = feature_extractor.compute_graph_concept_features(
                    test_loader,
                    concept_family=HasMotifFamily(cycle_lengths=(3, 4, 5, 6)),
                    auroc_thresh=0.6,
                    min_pos=20,
                    min_neg=20,
                )

            if sum(active_mask) >= 3:

                active_features = all_features[torch.tensor(active_mask).bool()]

                try:
                    si, _, er, er_raw = superposition_index(active_features, center_cols_for_effrank=False, norm_d=False)
                    d_a, mean_a, mu2_best_a, WNO_a, WNO_a_raw, _ = wno_ambient(active_features, remove_pc1=False)
                    d_i, mean_i, mu2_best_i, WNO_i, WNO_i_raw, pc1_e = wno_intrinsic(active_features, remove_pc1=False, effrank_for_r_center_cols=False)

                    feature_geometry = {
                        "all_features": all_features,
                        "active_mask": active_mask,
                        "active_count": sum(active_mask),

                        "auc": auc,

                        "si": si,
                        "er": er,
                        "er_raw": er_raw,

                        'pc1_e': pc1_e,

                        "d_a": d_a,
                        "mean_a": mean_a,
                        "mu2_best_a": mu2_best_a,
                        "WNO_a": WNO_a,
                        "WNO_a_raw": WNO_a_raw,

                        "d_i": d_i,
                        "mean_i": mean_i,
                        "mu2_best_i": mu2_best_i,
                        "WNO_i": WNO_i,
                        "WNO_i_raw": WNO_i_raw,
                    }

                    record[f"feature_geometry_{geometry}"] = feature_geometry
                except Exception as e:
                    print(e)
            else:
                self.logger.warning(f"Skipped for geometry {geometry} at s={self.seeds[experiment_idx]} because not enough active features.")

        return record

    def aggregate(self):

        pass


if __name__ == "__main__":

    import argparse, math, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seed-stop", type=int, default=111, help="exclusive upper bound")
    parser.add_argument("--group-idx", type=int, default=0, help="0-based index of the group to run")
    parser.add_argument("--group-size", type=int, default=0, help="0 => use all seeds in one group")
    parser.add_argument("--use-cached-models", action="store_true", help="whether to load cached models")
    args = parser.parse_args()

    logger = CustomLogger(level="DEBUG")

    all_seeds = list(range(args.seed_start, args.seed_stop))

    if args.group_size and args.group_size > 0:
        num_groups = math.ceil(len(all_seeds) / args.group_size)
        if args.group_idx < 0 or args.group_idx >= num_groups:
            logger.error(f"group_idx {args.group_idx} out of range [0, {num_groups - 1}]")
            sys.exit(2)
        left = args.group_idx * args.group_size
        right = min(left + args.group_size, len(all_seeds))
        seeds = all_seeds[left:right]
    else:
        seeds = all_seeds

    logger.info(f"Running group {args.group_idx} with seeds: {seeds}")

    for model_type, lr, weight_decay in [("GCN", 0.005, 1e-5)]:#, ("GIN", 0.001, 1e-5), ("GAT", 0.001, 1e-5)]:
        exp = ExperimentTopology(
            hidden_dim=16,
            pooling="mean",
            lr=lr,
            weight_decay=weight_decay,
            batch_size=256,
            model_type=model_type,
            seeds=seeds,
            device=Constants.DEVICE,
            logger=logger
        )

        exp.run_all(use_cached_models=args.use_cached_models)
