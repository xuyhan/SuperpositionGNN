from abc import ABC, abstractmethod
from typing import List, Dict, Any

import torch
from pathlib import Path
import random
import numpy as np

from src.logger import CustomLogger


class Constants:

    MODEL_SAVE_DIR = Path(__file__).resolve().parents[1] / "models"
    RESULT_SAVE_DIR = Path(__file__).resolve().parents[1] / "results"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Experiment(ABC):

    def __init__(self, seeds: List[int], device: torch.device, logger: CustomLogger):
        self.seeds = seeds
        self.device = device
        self.logger = logger

        self.save_path_dir = Constants.RESULT_SAVE_DIR / f"{type(self).__name__}"

    def _get_run_path(self, experiment_idx: int) -> Path:
        seed = self.seeds[experiment_idx]
        path_to_results = self.save_path_dir / self.summary() / f"{seed}.pt"
        return path_to_results

    def _load_results(self, experiment_idx: int) -> Dict[str, Any]:
        return torch.load(self._get_run_path(experiment_idx), weights_only=False)

    def _save_results(self, experiment_idx: int, results: Dict[str, Any]) -> None:
        path_to_save = self._get_run_path(experiment_idx)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving result to {path_to_save}.")
        torch.save(results, path_to_save)

    def run_all(self, use_cached_models: bool):

        for idx in range(len(self.seeds)):
            self.logger.info(f"Initiating experiment {idx} with seed {self.seeds[idx]}.")

            curr_seed = self.seeds[idx]
            torch.manual_seed(curr_seed)
            random.seed(curr_seed)
            np.random.seed(curr_seed)

            results = self.run_once(idx, use_cached_models)
            self._save_results(idx, results)

    @abstractmethod
    def summary(self) -> str:
        pass

    @abstractmethod
    def run_once(self, experiment_idx: int, use_cached_models: bool) -> Dict[str, Any]:
        pass

    @abstractmethod
    def aggregate(self):
        pass


