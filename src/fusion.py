"""
VoiceGuard — Fusion Engine
─────────────────────────────────────────────────────────────────────────
Implements Section 10 of the VoiceGuard project plan:

    deep_score  = XGBoost_A.predict(wav2vec2_embedding)   # 768-dim in
    bio_score   = XGBoost_B.predict(biological_features)  # 101-dim in (see extract_bio.py for exact count)
    final_score = 0.6 * deep_score + 0.4 * bio_score

Plus the sliding-window alert rule from Section 9:
    Keep the last 3 chunk scores. If 2 of the last 3 exceed the threshold,
    trigger an alert (reduces false alarms from single-chunk noise).

─────────────────────────────────────────────────────────────────────────
MODEL STATUS: voiceguard_a.json and voiceguard_b.json have not been
trained yet (confirmed by the user — same situation as the originally
referenced fusion_model.pkl, which was an empty placeholder file).

Until those real, trained XGBoost boosters are dropped into backend/models/,
FusionEngine runs in PLACEHOLDER mode: it returns a clearly-flagged,
heuristic, non-random score derived directly from real feature signal
(not Math.random()), so the pipeline is fully exercisable end-to-end, but
every API response marks placeholder results so the dashboard / frontend
can show this honestly rather than presenting fabricated confidence as
real model output.

THE MOMENT real voiceguard_a.json / voiceguard_b.json files are placed in
backend/models/, FusionEngine automatically loads and uses them — no other
code changes required.
"""

import os
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

DEEP_WEIGHT = 0.6
BIO_WEIGHT = 0.4

# Starting thresholds from the plan (Section 6). The plan explicitly notes
# these are illustrative only and should be replaced after plotting a real
# ROC curve on validation data — NOT left hardcoded in production.
THRESHOLD_SUSPICIOUS = 0.40
THRESHOLD_FRAUD = 0.65

SLIDING_WINDOW_SIZE = 3
SLIDING_WINDOW_REQUIRED = 2  # 2 of last 3 chunks must exceed threshold to alert

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_A_PATH = os.path.join(MODELS_DIR, "voiceguard_a.json")
MODEL_B_PATH = os.path.join(MODELS_DIR, "voiceguard_b.json")


def verdict_from_score(score: float) -> str:
    if score > THRESHOLD_FRAUD:
        return "FRAUD"
    if score > THRESHOLD_SUSPICIOUS:
        return "SUSPICIOUS"
    return "CLEAN"


def risk_level_from_score(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= THRESHOLD_FRAUD:
        return "HIGH"
    if score >= THRESHOLD_SUSPICIOUS:
        return "ELEVATED"
    return "LOW"


class FusionEngine:
    def __init__(self):
        self.model_a = None  # XGBoost A — deep stream (768-dim Wav2Vec2 embedding -> score)
        self.model_b = None  # XGBoost B — biological stream (101-dim -> score)
        self.using_real_models = False
        self.load_status = "not_attempted"

    def load(self):
        if not XGBOOST_AVAILABLE:
            self.load_status = "xgboost_not_installed"
            self.using_real_models = False
            return

        a_exists = os.path.exists(MODEL_A_PATH) and os.path.getsize(MODEL_A_PATH) > 0
        b_exists = os.path.exists(MODEL_B_PATH) and os.path.getsize(MODEL_B_PATH) > 0

        if not (a_exists and b_exists):
            self.load_status = (
                f"placeholder_mode (missing or empty: "
                f"{'voiceguard_a.json ' if not a_exists else ''}"
                f"{'voiceguard_b.json' if not b_exists else ''})".strip()
            )
            self.using_real_models = False
            return

        try:
            self.model_a = xgb.Booster()
            self.model_a.load_model(MODEL_A_PATH)
            self.model_b = xgb.Booster()
            self.model_b.load_model(MODEL_B_PATH)
            self.using_real_models = True
            self.load_status = "loaded"
        except Exception as e:
            self.using_real_models = False
            self.load_status = f"load_failed: {e}"

    # ── Stream predictions ──────────────────────────────────────────
    def predict_deep_score(self, embedding_768: np.ndarray) -> float:
        if self.using_real_models:
            dmat = xgb.DMatrix(embedding_768.reshape(1, -1))
            return float(self.model_a.predict(dmat)[0])
        return self._placeholder_deep_score(embedding_768)

    def predict_bio_score(self, bio_vector_102: np.ndarray) -> float:
        if self.using_real_models:
            dmat = xgb.DMatrix(bio_vector_102.reshape(1, -1))
            return float(self.model_b.predict(dmat)[0])
        return self._placeholder_bio_score(bio_vector_102)

    def fuse(self, deep_score: float, bio_score: float) -> float:
        return float(np.clip(DEEP_WEIGHT * deep_score + BIO_WEIGHT * bio_score, 0.0, 1.0))

    # ── Sliding-window alert logic (Section 9 of the plan) ──────────
    @staticmethod
    def check_sliding_window_alert(recent_scores: list[float], threshold: float = THRESHOLD_FRAUD) -> bool:
        """recent_scores should be the last up-to-3 chunk scores, oldest
        first. Returns True if >= SLIDING_WINDOW_REQUIRED of the last
        SLIDING_WINDOW_SIZE scores exceed threshold.
        """
        window = recent_scores[-SLIDING_WINDOW_SIZE:]
        exceed_count = sum(1 for s in window if s > threshold)
        return exceed_count >= SLIDING_WINDOW_REQUIRED

    # ── Placeholder heuristics (used ONLY until real .json models exist) ──
    # These are NOT Math.random() — they're deterministic, feature-derived
    # heuristics so the pipeline is testable end-to-end with honest,
    # reproducible (not fabricated) behavior. They are intentionally crude
    # and are clearly flagged via `using_real_models=False` in every API
    # response so nobody mistakes this for a trained model's output.
    @staticmethod
    def _placeholder_deep_score(embedding_768: np.ndarray) -> float:
        if embedding_768 is None or embedding_768.size == 0:
            return 0.5
        # Use embedding variance as a crude, deterministic proxy signal —
        # real speech embeddings from a genuine human voice tend to show
        # more frame-to-frame variance than embeddings of synthetic audio
        # pooled the same way. This is NOT a validated heuristic; it exists
        # only so placeholder mode produces varied, non-random, feature-
        # derived numbers instead of either a constant or np.random.
        norm = float(np.linalg.norm(embedding_768))
        score = 1.0 / (1.0 + np.exp(-(norm - 12.0) * 0.15))  # logistic squashing
        return float(np.clip(score, 0.02, 0.98))

    @staticmethod
    def _placeholder_bio_score(bio_vector_102: np.ndarray) -> float:
        if bio_vector_102 is None or bio_vector_102.size == 0:
            return 0.5
        # Jitter_Mean and Shimmer_Mean are indices 80 and 81 in FEATURE_ORDER
        # (after the 80 MFCC mean/std features). Very low jitter/shimmer is
        # the plan's own stated #1 biological fraud indicator (Section 4.4/4.5).
        jitter = bio_vector_102[80] if len(bio_vector_102) > 80 else 0.02
        shimmer = bio_vector_102[81] if len(bio_vector_102) > 81 else 0.02
        liveness = jitter + shimmer
        score = 1.0 / (1.0 + np.exp((liveness - 0.02) * 80))  # low liveness -> high score
        return float(np.clip(score, 0.02, 0.98))


# Module-level singleton — `.load()`-ed once by app.py at startup
fusion_engine = FusionEngine()
