from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Sequence


log = logging.getLogger(__name__)


class VoiceDependencyError(RuntimeError):
    pass


class OpenWakeWordDetector:
    def __init__(
        self,
        *,
        model_dir: Path,
        wakeword_model_file: str = "computer_v2.onnx",
        melspec_model_file: str = "melspectrogram.onnx",
        embedding_model_file: str = "embedding_model.onnx",
        wakeword_label: str = "computer_v2",
        threshold: float = 0.5,
    ) -> None:
        self.model_dir = model_dir
        self.wakeword_model_file = wakeword_model_file
        self.melspec_model_file = melspec_model_file
        self.embedding_model_file = embedding_model_file
        self.wakeword_label = wakeword_label
        self.threshold = threshold
        self._model: Any | None = None
        self._resolved_prediction_key: str | None = None
        self._warned_missing_label: bool = False

    @staticmethod
    def _normalize_label(label: str) -> str:
        # Compare labels by stem in lowercase so "computer_v2" and
        # "computer_v2.onnx" are treated as equivalent.
        return Path(label).stem.lower().strip()

    def _resolve_prediction_key(self, prediction: dict[str, Any]) -> str | None:
        if not prediction:
            return None

        if self._resolved_prediction_key in prediction:
            return self._resolved_prediction_key

        wanted = self._normalize_label(self.wakeword_label)
        for key in prediction.keys():
            if self._normalize_label(str(key)) == wanted:
                self._resolved_prediction_key = str(key)
                return self._resolved_prediction_key
        return None

    @staticmethod
    def _coerce_score(raw: Any) -> float:
        """Convert model output to a single score.

        openWakeWord output values can be Python floats, numpy scalars, or
        arrays/lists depending on backend/version. Use the max across values
        so we never silently drop valid scores.
        """
        try:
            import numpy as np

            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return 0.0
            return float(arr.max())
        except Exception:
            return 0.0

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from openwakeword.model import Model
        except Exception as exc:  # pragma: no cover
            raise VoiceDependencyError("openWakeWord is not installed") from exc

        wakeword_path = self.model_dir / self.wakeword_model_file
        melspec_path = self.model_dir / self.melspec_model_file
        embedding_path = self.model_dir / self.embedding_model_file

        if not wakeword_path.exists():
            raise VoiceDependencyError(f"Wake word model not found: {wakeword_path}")
        if not melspec_path.exists():
            raise VoiceDependencyError(f"openWakeWord melspec model not found: {melspec_path}")
        if not embedding_path.exists():
            raise VoiceDependencyError(f"openWakeWord embedding model not found: {embedding_path}")

        model_sig = inspect.signature(Model)
        wakeword_arg = "wakeword_models" if "wakeword_models" in model_sig.parameters else "wakeword_model_paths"
        full_kwargs: dict[str, Any] = {
            wakeword_arg: [str(wakeword_path)],
            "inference_framework": "onnx",
            "melspec_model_path": str(melspec_path),
            "embedding_model_path": str(embedding_path),
        }

        try:
            self._model = Model(**full_kwargs)
        except TypeError:
            # Older openWakeWord versions reject backend kwargs.
            self._model = Model(**{wakeword_arg: [str(wakeword_path)]})
        return self._model

    def reset(self) -> None:
        model = self._ensure_model()
        model.reset()

    def score(self, frame: Sequence[int]) -> float:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise VoiceDependencyError("numpy is not installed") from exc

        model = self._ensure_model()
        samples = np.asarray(frame, dtype=np.int16)
        prediction = model.predict(samples)
        log.debug("OWW raw prediction: %r", prediction)
        if isinstance(prediction, tuple):
            prediction = prediction[0] if prediction else {}
        if not isinstance(prediction, dict):
            prediction = {}

        key = self._resolve_prediction_key(prediction)
        if key is not None:
            raw = prediction.get(key, 0.0)
            return self._coerce_score(raw)

        # If model keys do not match configured label, use strongest score so
        # wake detection still functions while logging actionable diagnostics.
        best_key: str | None = None
        best_score = 0.0
        for pred_key, raw in prediction.items():
            score = self._coerce_score(raw)
            if best_key is None or score > best_score:
                best_key = str(pred_key)
                best_score = score

        if best_key is not None and not self._warned_missing_label:
            self._warned_missing_label = True
            log.warning(
                "OWW label '%s' not found in prediction keys %s; using strongest key '%s'",
                self.wakeword_label,
                list(prediction.keys()),
                best_key,
            )

        return best_score

    def triggered(self, frame: Sequence[int]) -> tuple[float, bool]:
        score = self.score(frame)
        if score >= self.threshold:
            self.reset()
            return score, True
        return score, False
