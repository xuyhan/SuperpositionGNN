from typing import List, Dict, Any

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from src.base_model import BaseModel
from src.data import SharedDataset
from src.features import FeatureExtractor
from src.features import IsTypeFamily, AdjacentToFamily, GraphLabelFamily
from src.logger import CustomLogger
from src.measure import superposition_index, wno_ambient, wno_intrinsic
from src.utils import Constants, compute_pos_weight_from_loader
from src.utils import Experiment


class ExperimentSuperposition(Experiment):

    def __init__(self, k: int, d_sweep: List[int], pooling: str, lr: float, batch_size: int, model_type: str,
                 seeds: list[int], device: torch.device, logger: CustomLogger):

        super().__init__(seeds, device, logger)

        self.k = k
        self.d_sweep = d_sweep
        self.pooling = pooling
        self.lr = lr
        self.batch_size = batch_size
        self.model_type = model_type

    def _create_and_fit_model(self, d: int, experiment_idx: int, train_loader: DataLoader,
                              use_cached_models: bool) -> BaseModel:

        pos_weight, _ = compute_pos_weight_from_loader(train_loader, self.k)

        model = BaseModel(identifier=f"Exp1_GNN={self.model_type}_pool={self.pooling}_k={self.k}_d={d}_s={self.seeds[experiment_idx]}",
                          model_type=self.model_type, in_dim=self.k,
                          hidden_dims=[self.k, d], out_dim=self.k,
                          num_ff=1,
                          pooling=self.pooling,
                          criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device), reduction='none'),
                          logger=self.logger)

        model.set_device(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        model.fit(train_loader, optimizer, num_epochs=200, use_cached=use_cached_models)

        return model

    def summary(self) -> str:

        return f"k={self.k},pool={self.pooling},lr={self.lr},batch={self.batch_size},type={self.model_type},freeze=False"

    def run_once(self, experiment_idx: int, use_cached_models: bool) -> Dict[str, Any]:

        if experiment_idx >= len(self.seeds):
            raise Exception(f"Experiment {experiment_idx} out of range")

        record = {}

        for d in self.d_sweep:

            train_loader = DataLoader(SharedDataset.PAIRWISE_16.train_data, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(SharedDataset.PAIRWISE_16.test_data, batch_size=self.batch_size, shuffle=False)

            model = self._create_and_fit_model(d, experiment_idx, train_loader, use_cached_models)

            avg_pure_embeddings, avg_pure_predictions, avg_loss, acc_total, prec_total, rec_total, auc_total, per_class_results = model.evaluate(
                test_loader)

            record[d] = {
                "loss": avg_loss,
                "acc": acc_total,
                "rec": rec_total,
                "per_class_results": per_class_results
            }

            feature_extractor = FeatureExtractor(model)

            self.logger.info(f"d={d} acc={acc_total:.3f} prec={prec_total:.3f} rec={rec_total:.3f}")

            for geometry in ["graph_centroid_gt", "graph_centroid_pr", "graph_centroid_in", "graph_concept_1", "graph_concept_2",
                             "node_concept_11", "node_concept_12", "node_concept_21", "node_concept_22"]:

                auc = None

                if geometry == "graph_centroid_gt":

                    all_features, _, _, _, active_mask = feature_extractor.compute_centroid_features(test_loader,
                                                                                                     centroid_mode="gt",
                                                                                                     one_hot_only=True,
                                                                                                     active_thresh=0.5)

                elif geometry == "graph_centroid_pr":

                    all_features, _, _, _, active_mask = feature_extractor.compute_centroid_features(test_loader,
                                                                                                     centroid_mode="pr",
                                                                                                     one_hot_only=True,
                                                                                                     min_n=50)

                elif geometry == "graph_centroid_in":

                    all_features, _, _, _, active_mask = feature_extractor.compute_centroid_features(test_loader,
                                                                                                     centroid_mode="in",
                                                                                                     one_hot_only=True,
                                                                                                     min_n=50)

                elif geometry == "graph_concept_1":

                    graph_concept_family = GraphLabelFamily(label_source='pred>0.5', one_hot_only=True)

                    all_features, _, auc, active_mask = feature_extractor.compute_graph_concept_features(
                        test_loader,
                        concept_family=graph_concept_family,
                        auroc_thresh=0.6,
                        min_pos=20,
                        min_neg=20,
                    )
                elif geometry == "graph_concept_2":

                    graph_concept_family = GraphLabelFamily(label_source='gt', one_hot_only=True)

                    all_features, _, auc, active_mask = feature_extractor.compute_graph_concept_features(
                        test_loader,
                        concept_family=graph_concept_family,
                        auroc_thresh=0.6,
                        min_pos=20,
                        min_neg=20
                    )
                elif geometry == "node_concept_11":

                    adj_family = AdjacentToFamily(threshold=0.5, symmetrize=True, ignore_self_loops=True)

                    all_features, _, auc, active_mask = feature_extractor.compute_node_concept_features(
                        test_loader,
                        concept_family=adj_family,
                        layer_idx=0,
                        pre_relu=False,
                        auroc_thresh=0.6,
                        min_pos=200,
                        min_neg=200,
                    )
                elif geometry == "node_concept_12":
                    adj_family = AdjacentToFamily(threshold=0.5, symmetrize=True, ignore_self_loops=True)

                    all_features, _, auc, active_mask = feature_extractor.compute_node_concept_features(
                        test_loader,
                        concept_family=adj_family,
                        layer_idx=1,
                        pre_relu=False,
                        auroc_thresh=0.6,
                        min_pos=200,
                        min_neg=200,
                    )
                elif geometry == "node_concept_21":

                    is_family = IsTypeFamily(threshold=0.5)

                    all_features, _, auc, active_mask = feature_extractor.compute_node_concept_features(
                        test_loader,
                        concept_family=is_family,
                        layer_idx=0,
                        pre_relu=False,
                        auroc_thresh=0.6,
                        min_pos=200,
                        min_neg=200,
                    )
                elif geometry == "node_concept_22":

                    is_family = IsTypeFamily(threshold=0.5)

                    all_features, _, auc, active_mask = feature_extractor.compute_node_concept_features(
                        test_loader,
                        concept_family=is_family,
                        layer_idx=1,
                        pre_relu=False,
                        auroc_thresh=0.6,
                        min_pos=200,
                        min_neg=200,
                    )
                else:
                    raise Exception(f"Unknown geometry {geometry}!")

                if all_features.shape[0] == 0:
                    self.logger.warning(f"No features found for geometry {geometry}!")
                    continue

                assert all_features.shape[0] == self.k
                active_features = all_features[torch.tensor(active_mask).bool()]

                if sum(active_mask) >= 3:

                    try:

                        is_centroid_geometry = "centroid" in geometry
                        is_graph_geometry = "graph" in geometry

                        si, _, er, er_raw = superposition_index(active_features, center_cols_for_effrank=is_centroid_geometry,
                                                                norm_d=False)
                        d_a, mean_a, mu2_best_a, WNO_a, WNO_a_raw, _ = wno_ambient(active_features,
                                                                                   remove_pc1=is_graph_geometry)
                        d_i, mean_i, mu2_best_i, WNO_i, WNO_i_raw, pc1_e = wno_intrinsic(active_features,
                                                                                     remove_pc1=is_graph_geometry,
                                                                                     effrank_for_r_center_cols=is_centroid_geometry)

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

                        record[d][f"feature_geometry_{geometry}"] = feature_geometry
                    except Exception as e:
                        print(e)
                else:
                    self.logger.warning(f"Skipped for geometry {geometry} at d={d} s={self.seeds[experiment_idx]} because not enough active features.")

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

    d_sweep = list(range(4, 17)) + list(range(18, 38, 2))

    logger.info(f"Running group {args.group_idx} with seeds: {seeds}")

    for model_type, lr in [("GCN", 0.1), ("GIN", 0.01), ("GAT", 0.01)]:
        exp = ExperimentSuperposition(
            k=16, d_sweep=d_sweep,
            pooling="mean", lr=lr, batch_size=256, model_type=model_type,
            seeds=seeds, device=Constants.DEVICE, logger=logger
        )

        exp.run_all(use_cached_models=args.use_cached_models)
