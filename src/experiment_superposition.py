from collections import defaultdict
import random
from typing import List, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from src.backbone import GNNModel
from src.data import SharedDataset
from src.geometry import structure_of_representation
from src.logger import CustomLogger
from src.measure import superposition_index, wno_ambient_pc1_removed, wno_intrinsic_pc1_removed
from src.utils import Constants
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

    def summary(self) -> str:

        return f"k={self.k},pool={self.pooling},lr={self.lr},batch={self.batch_size},type={self.model_type}"

    def run_once(self, experiment_idx: int, use_cached_models: bool) -> Dict[str, Any]:

        if experiment_idx >= len(self.seeds):
            raise Exception(f"Experiment {experiment_idx} out of range")

        record = {}

        for d in self.d_sweep:

            model = GNNModel(identifier=f"ExperimentSuperposition_d={d}_s={self.seeds[experiment_idx]}",
                             model_type=self.model_type, in_dim=self.k,
                             hidden_dims=[self.k, d], out_dim=self.k,
                             freeze_final=True,
                             pooling=self.pooling,
                             criterion=torch.nn.BCEWithLogitsLoss(reduction='none'))

            model.set_device(self.device)

            train_loader = DataLoader(SharedDataset.PAIRWISE_12.train_data, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(SharedDataset.PAIRWISE_12.test_data, batch_size=self.batch_size, shuffle=False)

            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            model.fit(train_loader, optimizer, num_epochs=300, use_weighting=True, importance=(15.0, 10.0),
                      use_cached=use_cached_models)

            avg_pure_embeddings, avg_pure_predictions, avg_loss, acc_total, prec_total, rec_total, auc_total, per_class_results = model.evaluate(test_loader)
            _, _, _, _, _, _, is_active = structure_of_representation(self.k, avg_pure_predictions, avg_pure_embeddings, avg_loss)

            vecs = torch.vstack(list(avg_pure_embeddings.values()))
            assert vecs.shape[0] == self.k and vecs.shape[1] == d

            vecs_active = vecs[torch.tensor(is_active).bool()]

            record[d] = {
                "loss": avg_loss,
                "acc": acc_total,
                "rec": rec_total,
                "per_class_results": per_class_results,
                "vecs": vecs,
                "vecs_active": vecs_active,
                "n_active": sum(is_active)
            }

            if sum(is_active) >= 3:
                si, si_raw, eff_com, eff_raw = superposition_index(vecs_active)
                d_a, mean_a, mu2_best_a, WNO_a, WNO_a_raw, pc1_e_a = wno_ambient_pc1_removed(vecs_active)
                d_i, mean_i, mu2_best_i, WNO_i, WNO_i_raw, pc1_e_i = wno_intrinsic_pc1_removed(vecs_active)

                self.logger.info(f"d={d} acc={acc_total:.3f} prec={prec_total:.3f} rec={rec_total:.3f}")
                self.logger.info(f"active features: {sum(is_active)}")

                record[d] |= {
                    "si": si,
                    "si_raw": si_raw,
                    "eff_com": eff_com,
                    "eff_raw": eff_raw,

                    "d_a": d_a,
                    "mean_a": mean_a,
                    "mu2_best_a": mu2_best_a,
                    "WNO_a": WNO_a,
                    "WNO_a_raw": WNO_a_raw,
                    "pc1_e_a": pc1_e_a,

                    "d_i": d_i,
                    "mean_i": mean_i,
                    "mu2_best_i": mu2_best_i,
                    "WNO_i": WNO_i,
                    "WNO_i_raw": WNO_i_raw,
                    "pc1_e_i": pc1_e_i,
                }

        return record

    def aggregate(self):

        aggregated_si = defaultdict(list)
        aggregated_wno_r = defaultdict(list)

        for exp_idx in range(len(self.seeds)):
            record = self._load_results(exp_idx)

            aggregated_si[record["d"]].append(record["si"])
            aggregated_wno_r[record["d"]].append(record["WNO_r"])


if __name__ == "__main__":

    logger = CustomLogger(level="DEBUG")

    # seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    seeds = [112, 113, 114, 115, 116, 117, 118]

    exp = ExperimentSuperposition(k=12, d_sweep=[4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36],
                                  pooling="mean", lr=0.1, batch_size=256, model_type="GCN",
                                  seeds=seeds, device=Constants.DEVICE, logger=logger)

    # exp = ExperimentSuperposition(k=12, d_sweep=[4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36],
    #                               pooling="mean", lr=0.1, batch_size=256, model_type="GCN",
    #                               seeds=[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111], device=Constants.DEVICE, logger=logger)
    #
    exp.run_all(use_cached_models=True)
