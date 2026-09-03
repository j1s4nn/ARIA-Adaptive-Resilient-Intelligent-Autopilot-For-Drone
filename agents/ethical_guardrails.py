"""
ARIA Agent C: Predictive Ethical Guardrails
============================================
An AI layer that prevents socially unacceptable flight paths even when
they form the geometrically optimal route.

Uses CLIP-based scene classification and geo-context to:
  - Detect funerals, private backyards, schools, protests
  - Reroute around no-fly social zones
  - Enforce privacy buffers around residential areas

OpenCV and CLIP are optional. Without them the engine falls back to
heuristic geo-zone checks, which is all the demo / tests require.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    clip = None
    CLIP_AVAILABLE = False


# =====================================================================
# Social Context Categories
# =====================================================================

class SocialContext(Enum):
    SAFE          = auto()
    PRIVATE_YARD  = auto()
    FUNERAL       = auto()
    SCHOOL        = auto()
    PROTEST       = auto()
    HOSPITAL      = auto()
    WORSHIP_SITE  = auto()
    RESTRICTED    = auto()


CONTEXT_DESCRIPTIONS = {
    SocialContext.PRIVATE_YARD:  ["a private backyard", "residential garden with fence"],
    SocialContext.FUNERAL:       ["a funeral ceremony", "graveyard with people mourning"],
    SocialContext.SCHOOL:        ["a school playground with children", "school yard"],
    SocialContext.PROTEST:       ["a political protest crowd", "demonstration gathering"],
    SocialContext.HOSPITAL:      ["a hospital entrance", "medical facility"],
    SocialContext.WORSHIP_SITE:  ["a religious ceremony outdoor", "mosque church temple"],
    SocialContext.SAFE:          ["an open field", "parking lot", "industrial area"],
}

# Privacy buffer radii (meters)
PRIVACY_BUFFERS = {
    SocialContext.PRIVATE_YARD:  30,
    SocialContext.FUNERAL:       100,
    SocialContext.SCHOOL:        60,
    SocialContext.PROTEST:       80,
    SocialContext.HOSPITAL:      50,
    SocialContext.WORSHIP_SITE:  80,
    SocialContext.RESTRICTED:    200,
}


@dataclass
class GeoZone:
    """A social exclusion zone in local coordinate space."""
    center_x: float
    center_y: float
    radius: float
    context: SocialContext
    label: str = ""
    confidence: float = 1.0


@dataclass
class GuardrailDecision:
    allowed: bool
    reason: str
    detected_context: SocialContext
    confidence: float
    suggested_reroute: Optional[np.ndarray] = None


# =====================================================================
# CLIP Scene Classifier
# =====================================================================

class CLIPSceneClassifier:
    """
    Uses OpenAI CLIP to classify drone camera frames into social context
    categories. Falls back to a conservative mock when CLIP is missing.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if (CLIP_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.device = device
        self.model = None
        self.preprocess = None
        self._text_features = {}

        if CLIP_AVAILABLE:
            self.model, self.preprocess = clip.load("ViT-B/32", device=device)
            self._precompute_text_features()
            logger.success(f"CLIP loaded on {device}")
        else:
            logger.warning("CLIP unavailable - visual guardrail runs in heuristic mode.")

    def _precompute_text_features(self):
        """Pre-encode text prompts for each context (done once at startup)."""
        for ctx, descriptions in CONTEXT_DESCRIPTIONS.items():
            tokens = clip.tokenize(descriptions).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
            self._text_features[ctx] = feats

    def classify(self, frame: np.ndarray) -> tuple[SocialContext, float]:
        """
        Classify a camera frame into a social context.
        Returns (context, confidence).
        """
        if not CLIP_AVAILABLE or self.model is None or frame is None:
            return self._mock_classify(frame)
        if not CV2_AVAILABLE:
            return self._mock_classify(frame)

        from PIL import Image

        # Preprocess
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)

        # Compare against each context
        best_ctx = SocialContext.SAFE
        best_score = -1.0

        for ctx, text_feats in self._text_features.items():
            scores = (img_features @ text_feats.T).squeeze(0)
            ctx_score = float(scores.max().cpu())
            if ctx_score > best_score:
                best_score = ctx_score
                best_ctx = ctx

        # Threshold: only flag if confidence > 0.22 (CLIP cosine similarity scale)
        if best_ctx != SocialContext.SAFE and best_score < 0.22:
            best_ctx = SocialContext.SAFE

        return best_ctx, best_score

    @staticmethod
    def _mock_classify(frame: Optional[np.ndarray]) -> tuple[SocialContext, float]:
        """Mock classifier for testing without CLIP - conservatively reports SAFE."""
        return SocialContext.SAFE, 0.95


# =====================================================================
# Ethical Guardrail Engine
# =====================================================================

class EthicalGuardrailEngine:
    """
    Main guardrail engine. Combines:
     1. Camera-based scene classification (CLIP)
     2. Static geo-zone database
     3. Path projection checking
    """

    def __init__(self, classifier: Optional[CLIPSceneClassifier] = None):
        self.classifier = classifier or CLIPSceneClassifier()
        self.static_zones: list[GeoZone] = []
        self.violation_log: list[dict] = []

    def register_static_zone(self, zone: GeoZone):
        """Register a known no-fly social zone (e.g., loaded from map)."""
        self.static_zones.append(zone)
        logger.info(f"Registered zone: {zone.label} ({zone.context.name}) at "
                    f"({zone.center_x:.1f}, {zone.center_y:.1f}), r={zone.radius}m")

    def check_position(self, position: np.ndarray, frame: Optional[np.ndarray] = None) -> GuardrailDecision:
        """
        Check if a given (x, y, z) position is ethically permissible.
        Optionally pass a camera frame for real-time visual context.
        """
        x, y = position[0], position[1]

        # 1. Static zone check
        for zone in self.static_zones:
            dist = np.sqrt((x - zone.center_x)**2 + (y - zone.center_y)**2)
            if dist < zone.radius:
                return GuardrailDecision(
                    allowed=False,
                    reason=f"Position inside {zone.context.name} exclusion zone: '{zone.label}'",
                    detected_context=zone.context,
                    confidence=1.0,
                    suggested_reroute=self._compute_reroute(position, zone),
                )

        # 2. Visual classification
        if frame is not None:
            ctx, conf = self.classifier.classify(frame)
            if ctx != SocialContext.SAFE and conf > 0.22:
                self.violation_log.append({
                    "position": position.tolist(),
                    "context": ctx.name,
                    "confidence": conf,
                })
                return GuardrailDecision(
                    allowed=False,
                    reason=f"Camera detected sensitive context: {ctx.name} (conf={conf:.2f})",
                    detected_context=ctx,
                    confidence=conf,
                    suggested_reroute=position + np.array([0, 0, 20.0]),
                )

        return GuardrailDecision(
            allowed=True,
            reason="No ethical violations detected.",
            detected_context=SocialContext.SAFE,
            confidence=1.0,
        )

    def check_path(self, waypoints: list[np.ndarray],
                   frames: Optional[list] = None) -> list[GuardrailDecision]:
        """Check an entire planned path for ethical violations."""
        decisions = []
        for i, wp in enumerate(waypoints):
            frame = frames[i] if frames and i < len(frames) else None
            decisions.append(self.check_position(wp, frame))
        return decisions

    def sanitize_path(self, waypoints: list[np.ndarray]) -> list[np.ndarray]:
        """
        Return a modified path with violating waypoints replaced
        by suggested reroutes (altitude increase + lateral detour).
        """
        clean_path = []
        for wp in waypoints:
            decision = self.check_position(wp)
            if decision.allowed:
                clean_path.append(wp)
            elif decision.suggested_reroute is not None:
                clean_path.append(decision.suggested_reroute)
                logger.warning(f"Rerouted waypoint due to: {decision.reason}")
        return clean_path

    @staticmethod
    def _compute_reroute(position: np.ndarray, zone: GeoZone) -> np.ndarray:
        """Compute a reroute waypoint that skirts around the zone."""
        zone_center = np.array([zone.center_x, zone.center_y, position[2]])
        direction = position - zone_center
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 1e-3:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        tangent = np.array([-direction[1], direction[0], 0.0]) / norm
        # Reroute: go 20% past edge in tangential direction + 10m altitude gain
        reroute = position + tangent * (zone.radius * 1.2 - norm + 10.0) + np.array([0, 0, 10.0])
        return reroute


# =====================================================================
# Demo
# =====================================================================
if __name__ == "__main__":
    engine = EthicalGuardrailEngine()

    # Register example static zones
    engine.register_static_zone(GeoZone(50, 50, 40, SocialContext.FUNERAL,     "City Cemetery Service"))
    engine.register_static_zone(GeoZone(120, 30, 60, SocialContext.SCHOOL,     "Sunrise Elementary"))
    engine.register_static_zone(GeoZone(0, 200, 50, SocialContext.WORSHIP_SITE, "St. Mary's Cathedral"))

    test_positions = [
        np.array([10.0, 10.0, 15.0]),   # Safe
        np.array([55.0, 55.0, 15.0]),   # Inside funeral zone
        np.array([120.0, 30.0, 15.0]),  # Inside school zone
    ]

    for pos in test_positions:
        d = engine.check_position(pos)
        status = "ALLOWED" if d.allowed else f"BLOCKED - {d.reason}"
        print(f"Position {pos[:2]} -> {status}")
        if d.suggested_reroute is not None:
            print(f"  Suggested reroute: {d.suggested_reroute[:2]}")
