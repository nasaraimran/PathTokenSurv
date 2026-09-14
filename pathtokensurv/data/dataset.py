from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from pathtokensurv.data.preprocessing import ProcessedCohort


class MultiModalSurvivalDataset(Dataset):
    def __init__(self, cohort: ProcessedCohort) -> None:
        self.cohort = cohort

    def __len__(self) -> int:
        return len(self.cohort)

    def __getitem__(self, index: int) -> Dict:
        return {
            "patient_id": str(self.cohort.patient_ids[index]),
            "molecular": {
                name: tensor[index]
                for name, tensor in self.cohort.molecular.items()
            },
            "clinical_continuous": self.cohort.clinical_continuous[index],
            "clinical_categorical": {
                name: tensor[index]
                for name, tensor in self.cohort.clinical_categorical.items()
            },
            "availability": self.cohort.availability[index],
            "time": self.cohort.times[index],
            "event": self.cohort.events[index],
            "cancer": self.cohort.cancers[index],
        }
