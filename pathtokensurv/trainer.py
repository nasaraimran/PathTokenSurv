from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from pathtokensurv.config import TrainingConfig
from pathtokensurv.data.time import TimeDiscretizer
from pathtokensurv.losses import discrete_survival_loss, total_training_loss
from pathtokensurv.metrics import evaluate_survival_predictions
from pathtokensurv.models.model import ModalityMaskSampler, PathTokenSurv
from pathtokensurv.plotting import plot_training_history
from pathtokensurv.utils.device import move_batch_to_device
from pathtokensurv.utils.io import save_json


@dataclass
class PredictionBundle:
    patient_ids: List[str]
    times: np.ndarray
    events: np.ndarray
    cancers: np.ndarray
    survival: np.ndarray
    fused: np.ndarray


class Trainer:
    def __init__(
        self,
        model: PathTokenSurv,
        discretizer: TimeDiscretizer,
        config: TrainingConfig,
        device: torch.device,
        output_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.discretizer = discretizer
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.scheduler_t_max),
            eta_min=config.min_learning_rate,
        )
        self.mask_sampler = ModalityMaskSampler(
            config.modality_drop_probability,
            always_keep_clinical=config.always_keep_clinical,
        )

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {
            "total": 0.0,
            "survival": 0.0,
            "reconstruction": 0.0,
            "consistency": 0.0,
            "baseline": 0.0,
        }
        count = 0
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model.forward_training(batch, self.mask_sampler)
            losses = total_training_loss(
                model=self.model,
                output=output,
                batch=batch,
                discretizer=self.discretizer,
                lambda_reconstruction=(
                    self.config.lambda_reconstruction if self.model.config.use_reconstruction else 0.0
                ),
                lambda_consistency=(
                    self.config.lambda_consistency if self.model.config.use_subset_consistency else 0.0
                ),
                lambda_baseline=self.config.lambda_baseline,
                baseline_smoothness=self.config.baseline_smoothness,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            self.optimizer.step()
            batch_size = int(batch["time"].shape[0])
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_size
            count += batch_size
        self.scheduler.step()
        return {key: value / max(count, 1) for key, value in totals.items()}

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> PredictionBundle:
        self.model.eval()
        patient_ids: List[str] = []
        times: List[np.ndarray] = []
        events: List[np.ndarray] = []
        cancers: List[np.ndarray] = []
        survival: List[np.ndarray] = []
        fused: List[np.ndarray] = []
        for batch in loader:
            patient_ids.extend(list(batch["patient_id"]))
            batch = move_batch_to_device(batch, self.device)
            output = self.model(batch)
            times.append(batch["time"].detach().cpu().numpy())
            events.append(batch["event"].detach().cpu().numpy())
            cancers.append(batch["cancer"].detach().cpu().numpy())
            survival.append(output["survival"].detach().cpu().numpy())
            fused.append(output["fused"].detach().cpu().numpy())
        return PredictionBundle(
            patient_ids=patient_ids,
            times=np.concatenate(times),
            events=np.concatenate(events),
            cancers=np.concatenate(cancers),
            survival=np.concatenate(survival),
            fused=np.concatenate(fused),
        )

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> tuple[float, PredictionBundle]:
        """Compute validation loss and predictions in one model pass.

        v1.5.5 evaluated the validation set twice per epoch: once for survival
        loss and again for C-index/IBS predictions. This method preserves the
        exact metrics but removes the duplicate forward pass.
        """
        self.model.eval()
        total = 0.0
        count = 0
        patient_ids: List[str] = []
        times: List[np.ndarray] = []
        events: List[np.ndarray] = []
        cancers: List[np.ndarray] = []
        survival: List[np.ndarray] = []
        fused: List[np.ndarray] = []
        for batch in loader:
            patient_ids.extend(list(batch["patient_id"]))
            batch = move_batch_to_device(batch, self.device)
            output = self.model(batch)
            loss = discrete_survival_loss(
                output["logits"], batch["time"], batch["event"], self.discretizer
            )
            batch_size = int(batch["time"].shape[0])
            total += float(loss.cpu()) * batch_size
            count += batch_size
            times.append(batch["time"].detach().cpu().numpy())
            events.append(batch["event"].detach().cpu().numpy())
            cancers.append(batch["cancer"].detach().cpu().numpy())
            survival.append(output["survival"].detach().cpu().numpy())
            fused.append(output["fused"].detach().cpu().numpy())
        bundle = PredictionBundle(
            patient_ids=patient_ids,
            times=np.concatenate(times),
            events=np.concatenate(events),
            cancers=np.concatenate(cancers),
            survival=np.concatenate(survival),
            fused=np.concatenate(fused),
        )
        return total / max(count, 1), bundle

    @torch.no_grad()
    def validation_loss(self, loader: DataLoader) -> float:
        # Retained for API compatibility. New training uses evaluate_loader().
        loss, _ = self.evaluate_loader(loader)
        return loss

    def _align_scheduler_after_legacy_resume(self, epoch: int) -> None:
        """Align CosineAnnealingLR when loading a v1.5.5 checkpoint."""
        self.scheduler.last_epoch = int(epoch)
        self.scheduler._step_count = int(epoch) + 1
        self.scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def load_training_checkpoint(self, path: str | Path) -> dict:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        else:
            self._align_scheduler_after_legacy_resume(int(checkpoint.get("epoch", 0)))
        return checkpoint

    def _write_partial_history(self, history_rows: list[dict]) -> None:
        history = pd.DataFrame(history_rows)
        history.to_csv(self.output_dir / "training_history.csv", index=False)
        if not history.empty:
            plot_training_history(history, self.output_dir / "training_history.png")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_times: np.ndarray,
        train_events: np.ndarray,
        resume_from: str | Path | None = None,
    ) -> pd.DataFrame:
        best_score = -np.inf
        best_epoch = -1
        wait = 0
        start_epoch = 1
        history_rows: list[dict] = []
        started = time.time()

        if resume_from is not None:
            checkpoint = self.load_training_checkpoint(resume_from)
            checkpoint_epoch = int(checkpoint.get("epoch", 0))
            best_epoch = int(checkpoint.get("best_epoch", checkpoint_epoch))
            best_score = float(checkpoint.get("best_score", checkpoint.get("score", -np.inf)))
            wait = int(checkpoint.get("wait", 0))
            start_epoch = checkpoint_epoch + 1
            history_path = self.output_dir / "training_history.csv"
            if history_path.exists():
                prior = pd.read_csv(history_path)
                history_rows = prior.to_dict(orient="records")
            print(
                f"Resuming from {resume_from}: checkpoint epoch={checkpoint_epoch}, "
                f"best epoch={best_epoch}, best score={best_score:.6f}; "
                f"next epoch={start_epoch}."
            )

        try:
            for epoch in range(start_epoch, self.config.max_epochs + 1):
                epoch_started = time.time()
                train_started = time.time()
                train_stats = self.train_epoch(train_loader)
                train_seconds = time.time() - train_started

                val_started = time.time()
                val_loss, val_predictions = self.evaluate_loader(val_loader)
                val_metrics = evaluate_survival_predictions(
                    train_times=train_times,
                    train_events=train_events,
                    test_times=val_predictions.times,
                    test_events=val_predictions.events,
                    survival=val_predictions.survival,
                    horizons=self.discretizer.right_edges,
                    calibration_bins=8,
                )
                val_seconds = time.time() - val_started
                c_index = val_metrics.get("c_index", float("nan"))
                ibs = val_metrics.get("ibs", float("nan"))
                score = (-val_loss) if not np.isfinite(c_index) else c_index - (ibs if np.isfinite(ibs) else 0.0)
                epoch_seconds = time.time() - epoch_started
                row = {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_stats.items()},
                    "val_survival_loss": val_loss,
                    "val_c_index": c_index,
                    "val_ibs": ibs,
                    "selection_score": score,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train_seconds": train_seconds,
                    "validation_seconds": val_seconds,
                    "epoch_seconds": epoch_seconds,
                }
                history_rows.append(row)
                print(
                    f"Epoch {epoch:03d} | train={train_stats['total']:.4f} "
                    f"val={val_loss:.4f} c-index={c_index:.4f} ibs={ibs:.4f} "
                    f"| {epoch_seconds/60.0:.1f} min "
                    f"(train {train_seconds/60.0:.1f}, val {val_seconds/60.0:.1f})"
                )

                if score > best_score + 1e-6:
                    best_score = score
                    best_epoch = epoch
                    wait = 0
                    self.save_checkpoint(
                        self.output_dir / "best_model.pt",
                        epoch,
                        score,
                        best_epoch=best_epoch,
                        best_score=best_score,
                        wait=wait,
                    )
                else:
                    wait += 1

                # Persist the current training state and history every epoch so an
                # interrupted multi-day run can be resumed exactly from the last
                # completed epoch.
                self.save_checkpoint(
                    self.output_dir / "last_model.pt",
                    epoch,
                    score,
                    best_epoch=best_epoch,
                    best_score=best_score,
                    wait=wait,
                )
                self._write_partial_history(history_rows)
                save_json(
                    {
                        "status": "RUNNING",
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "last_completed_epoch": epoch,
                        "training_seconds": time.time() - started,
                    },
                    self.output_dir / "training_summary.json",
                )

                if wait >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                    break
        except KeyboardInterrupt:
            self._write_partial_history(history_rows)
            save_json(
                {
                    "status": "INTERRUPTED",
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "last_completed_epoch": history_rows[-1]["epoch"] if history_rows else start_epoch - 1,
                    "training_seconds": time.time() - started,
                },
                self.output_dir / "training_summary.json",
            )
            raise

        history = pd.DataFrame(history_rows)
        self._write_partial_history(history_rows)
        elapsed = time.time() - started
        save_json(
            {
                "status": "PASS",
                "best_epoch": best_epoch,
                "best_score": best_score,
                "training_seconds": elapsed,
            },
            self.output_dir / "training_summary.json",
        )
        self.load_checkpoint(self.output_dir / "best_model.pt")
        return history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        score: float,
        *,
        best_epoch: int | None = None,
        best_score: float | None = None,
        wait: int = 0,
    ) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "epoch": int(epoch),
                "score": float(score),
                "best_epoch": int(epoch if best_epoch is None else best_epoch),
                "best_score": float(score if best_score is None else best_score),
                "wait": int(wait),
            },
            Path(path),
        )

    def load_checkpoint(self, path: str | Path) -> dict:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        return checkpoint
