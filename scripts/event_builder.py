#!/usr/bin/env python3
"""
Semantic Event Builder — replaces fixed-width chunking with signal-weighted
variable-length events.

Architecture
------------
Plugin-style signal stack: add a new ``BaseSignal`` subclass to
``DEFAULT_SIGNAL_STACK`` and it is automatically picked up by ``build_events()``.

Signal weights (additive per boundary candidate):
  scene_cut           5
  topic_shift         5
  speaker_change      5   (was 4 — equal to topic_shift, both mark clear transitions)
  long_silence        4   (was 3 — strong editorial signal)
  audio_energy        4   (reserved — caller supplies via extra_signals)
  transcript_density  3   (new — high WPM windows = content-rich moments)
  rapid_motion        2
  ocr_change          2   (reserved)
  face_enter          2   (reserved)
  music               3   (reserved)

Boundary fusion window: signals within 1.5 s count toward the same boundary.

Score thresholds:
  < 3   → not a real boundary (merged with neighbour)
  3-4   → LOW profile
  5-8   → MEDIUM profile
  ≥ 9   → HIGH profile

Processing profiles (control the full analysis strategy, not just token count):
  LOW    — 512 tok  | few frames    | quick template    | minimal reasoning
  MEDIUM — 1500 tok | moderate frames | rich template   | standard reasoning
  HIGH   — 4096 tok | many frames   | full template     | deep reasoning

Every event is analyzed by the VLM. Profile controls reasoning depth.
No events are skipped.

TODO: Replace TF-IDF cosine in ``TopicShiftSignal`` with sentence-transformers
      embeddings once that dependency is available in the runtime environment.
      TF-IDF misfires on vocab-dense technical topics where the same domain
      words appear across genuinely different sub-topics (e.g. "GPU memory cache"
      and "GPU KV cache utilization" score as similar despite topic shift).

Usage
-----
    from event_builder import build_events, events_to_chunks, event_stats, PROFILES

    events = build_events(
        duration=3600.0,
        scene_cuts=[0.0, 45.2, 112.7, ...],
        transcript_segments=[{start, end, text}, ...],
        extra_signals=[],
        max_event_s=36.0,
        min_event_s=5.0,
    )
    chunks = events_to_chunks(events, duration=3600.0, overlap_s=3.0)
    print(event_stats(events))
"""
from __future__ import annotations

import abc
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, NamedTuple

if TYPE_CHECKING:
    from chunk_dispatch import Chunk  # type: ignore[import]

# ── Processing profiles ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessingProfile:
    """Full analysis strategy for one tier of event importance."""
    name:            str
    max_tokens:      int
    frames_hint:     str   # "few" | "moderate" | "many"  (passed to prompt builder)
    prompt_template: str   # "quick" | "rich" | "full"
    reasoning_depth: str   # "minimal" | "standard" | "deep"


PROFILES: dict[str, ProcessingProfile] = {
    "LOW": ProcessingProfile(
        name="LOW",
        max_tokens=512,
        frames_hint="few",
        prompt_template="quick",
        reasoning_depth="minimal",
    ),
    "MEDIUM": ProcessingProfile(
        name="MEDIUM",
        max_tokens=1500,
        frames_hint="moderate",
        prompt_template="rich",
        reasoning_depth="standard",
    ),
    "HIGH": ProcessingProfile(
        name="HIGH",
        max_tokens=4096,
        frames_hint="many",
        prompt_template="full",
        reasoning_depth="deep",
    ),
}

# ── Score thresholds ──────────────────────────────────────────────────────────

MERGE_THRESHOLD = 3   # boundary score below this → merge with neighbour
MED_THRESHOLD   = 4   # score ≥ this → MEDIUM profile
HIGH_THRESHOLD  = 7   # score ≥ this → HIGH profile (lowered: scene_cuts=None caps max score; restore to 9 when scene cuts are wired)

# Signal fusion: signals within this many seconds cluster into one boundary
FUSION_WINDOW_S  = 1.5

# Speaker/silence gap thresholds
SPEAKER_GAP_S = 0.5
SILENCE_GAP_S = 2.0

# Topic-shift window parameters
TOPIC_WIN_WORDS  = 25
TOPIC_STEP_WORDS = 10
TOPIC_SIM_THRESH = 0.35

# ── SemanticEvent ─────────────────────────────────────────────────────────────

class SemanticEvent(NamedTuple):
    """One variable-length event for VLM analysis."""
    event_id:   int
    start:      float
    end:        float
    score:      int          # raw cumulative boundary score
    importance: float        # normalised [0, 1] across all events in video
    profile:    str          # "LOW" | "MEDIUM" | "HIGH"
    max_tokens: int          # VLM output token budget
    signals:    dict         # {signal_name: True} for each signal that fired

    @property
    def duration(self) -> float:
        return self.end - self.start


# ── Plugin signal interface ───────────────────────────────────────────────────

class BaseSignal(abc.ABC):
    """
    Plugin base for all signal extractors.

    Subclass, set ``name`` and ``weight``, implement ``extract()``, then
    append an instance to ``DEFAULT_SIGNAL_STACK``.
    """
    name:   str
    weight: int

    @abc.abstractmethod
    def extract(
        self,
        duration:      float,
        scene_cuts:    list[float],
        segments:      list[dict],
        **kwargs,
    ) -> list[tuple[float, str]]:
        """Return list of (timestamp_seconds, signal_name) pairs."""
        ...


# ── Concrete signal implementations ──────────────────────────────────────────

class SceneCutSignal(BaseSignal):
    name   = "scene_cut"
    weight = 5

    def extract(self, duration, scene_cuts, segments, **kwargs):
        return [(t, self.name) for t in scene_cuts if 0.0 < t < duration]


class SpeakerChangeSignal(BaseSignal):
    name   = "speaker_change"
    weight = 5

    def extract(self, duration, scene_cuts, segments, **kwargs):
        out: list[tuple[float, str]] = []
        prev_end = 0.0
        for seg in segments:
            start = float(seg.get("start", 0))
            gap   = start - prev_end
            # Gap is speaker-change-sized (not big enough to be silence)
            if SILENCE_GAP_S > gap >= SPEAKER_GAP_S:
                out.append((start, self.name))
            prev_end = float(seg.get("end", start))
        return out


class SilenceSignal(BaseSignal):
    name   = "long_silence"
    weight = 4

    def extract(self, duration, scene_cuts, segments, **kwargs):
        out: list[tuple[float, str]] = []
        prev_end = 0.0
        for seg in segments:
            start = float(seg.get("start", 0))
            if start - prev_end >= SILENCE_GAP_S:
                out.append((start, self.name))
            prev_end = float(seg.get("end", start))
        return out


class TopicShiftSignal(BaseSignal):
    # TODO: Replace with sentence-transformers embeddings when available.
    # TF-IDF misfires on vocab-dense technical content where the same domain
    # words repeat across different sub-topics (see module docstring).
    name   = "topic_shift"
    weight = 5

    def extract(self, duration, scene_cuts, segments, **kwargs):
        return [(t, self.name) for t in _topic_shift_times(segments)]


class TranscriptDensitySignal(BaseSignal):
    """High word-density windows flag content-rich moments worth deeper analysis."""
    name        = "transcript_density"
    weight      = 3
    WINDOW_S    = 10.0
    HIGH_WPM    = 120   # words per minute; above this → dense

    def extract(self, duration, scene_cuts, segments, **kwargs):
        if not segments:
            return []
        out: list[tuple[float, str]] = []
        bins: dict[int, float] = {}
        for seg in segments:
            text = seg.get("text", "")
            span = float(seg.get("end", 0)) - float(seg.get("start", 0))
            if span < 0.1:
                continue
            wps = len(text.split()) / span
            bin_idx = int(float(seg.get("start", 0)) / self.WINDOW_S)
            bins[bin_idx] = bins.get(bin_idx, 0.0) + wps * self.WINDOW_S
        threshold = self.HIGH_WPM * self.WINDOW_S / 60.0
        for bin_idx, word_count in bins.items():
            if word_count >= threshold:
                t = bin_idx * self.WINDOW_S
                if 0.0 < t < duration:
                    out.append((t, self.name))
        return out


class RapidMotionSignal(BaseSignal):
    """Burst of ≥ MIN_CUTS scene cuts within BURST_WINDOW_S seconds = rapid editing."""
    name          = "rapid_motion"
    weight        = 2
    BURST_WINDOW_S = 5.0
    MIN_CUTS       = 3

    def extract(self, duration, scene_cuts, segments, **kwargs):
        raw: list[tuple[float, str]] = []
        n = len(scene_cuts)
        for i in range(n):
            j = i + 1
            while j < n and scene_cuts[j] - scene_cuts[i] <= self.BURST_WINDOW_S:
                j += 1
            if j - i >= self.MIN_CUTS:
                raw.append((scene_cuts[i], self.name))
        # Deduplicate within 2 s
        deduped: list[tuple[float, str]] = []
        for item in sorted(raw):
            if not deduped or item[0] - deduped[-1][0] > 2.0:
                deduped.append(item)
        return deduped


# Default signal stack — add here to extend; no other code changes needed.
DEFAULT_SIGNAL_STACK: list[BaseSignal] = [
    SceneCutSignal(),
    SpeakerChangeSignal(),
    SilenceSignal(),
    TopicShiftSignal(),
    TranscriptDensitySignal(),
    RapidMotionSignal(),
    # Future additions (plug in by appending an instance):
    # AudioEnergySignal()   — librosa RMS peaks (weight 4)
    # MusicSignal()         — librosa onset + chroma (weight 3)
    # OCRSignal()           — Tesseract / EasyOCR frame diff (weight 2)
    # FaceEnterSignal()     — face-detection frame diff (weight 2)
]


# ── TF-IDF cosine similarity (zero-dep) ──────────────────────────────────────

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need dare ought used "
    "i me my myself we our ours ourselves you your yours he him his "
    "she her hers it its they them their what which who this that "
    "these those am at by for in of on to up as if so but or and not "
    "with from about into through during before after above below between "
    "out off over under again then once here there when where why how all "
    "each few more most other some such no nor only same than too very just "
    "s t don doesn isn wasn weren won wouldn".split()
)
_TOKEN_RE = re.compile(r"[a-z]+")


def _tokenise(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower())
            if w not in _STOPWORDS and len(w) > 2]


def _tf(tokens: list[str]) -> dict[str, float]:
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    n = len(tokens) or 1
    return {w: c / n for w, c in freq.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys  = set(a) | set(b)
    dot   = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    mag_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (mag_a * mag_b)


def _topic_shift_times(segments: list[dict]) -> list[float]:
    """Return timestamps where TF-IDF cosine distance exceeds TOPIC_SIM_THRESH."""
    word_times: list[tuple[float, str]] = []
    for seg in segments:
        text   = seg.get("text", "")
        start  = float(seg.get("start", 0))
        end    = float(seg.get("end", start + 0.01))
        tokens = _tokenise(text)
        if not tokens:
            continue
        step = (end - start) / len(tokens)
        for i, tok in enumerate(tokens):
            word_times.append((start + i * step, tok))

    if len(word_times) < TOPIC_WIN_WORDS * 2:
        return []

    shifts: list[float] = []
    i = 0
    while i + TOPIC_WIN_WORDS * 2 <= len(word_times):
        win_a = [w for _, w in word_times[i : i + TOPIC_WIN_WORDS]]
        win_b = [w for _, w in word_times[i + TOPIC_WIN_WORDS : i + TOPIC_WIN_WORDS * 2]]
        if 1 - _cosine(_tf(win_a), _tf(win_b)) > TOPIC_SIM_THRESH:
            shifts.append(word_times[i + TOPIC_WIN_WORDS][0])
        i += TOPIC_STEP_WORDS

    # Deduplicate: keep only one per 3-second window
    deduped: list[float] = []
    for t in sorted(shifts):
        if not deduped or t - deduped[-1] > 3.0:
            deduped.append(t)
    return deduped


# ── Boundary fusion ───────────────────────────────────────────────────────────

@dataclass
class _Boundary:
    time:    float
    score:   int            = 0
    signals: list[str]      = field(default_factory=list)


def _fuse_signals(
    all_signals: list[tuple[float, str]],
    signal_weights: dict[str, int],
    fusion_window: float = FUSION_WINDOW_S,
) -> list[_Boundary]:
    """Cluster (time, kind) pairs within fusion_window → scored _Boundary list."""
    if not all_signals:
        return []
    sorted_sigs = sorted(all_signals, key=lambda s: s[0])
    clusters: list[_Boundary] = []
    cur: _Boundary | None = None
    for t, kind in sorted_sigs:
        if cur is None or t - cur.time > fusion_window:
            cur = _Boundary(time=t)
            clusters.append(cur)
        cur.score += signal_weights.get(kind, 1)
        if kind not in cur.signals:
            cur.signals.append(kind)
    return clusters


# ── Span manipulation helpers ─────────────────────────────────────────────────

def _pairs(times: list[float]) -> Iterator[tuple[float, float]]:
    for i in range(len(times) - 1):
        yield times[i], times[i + 1]


def _collapse_short(times: list[float], min_s: float) -> list[float]:
    """Remove boundaries that create spans shorter than min_s."""
    if len(times) < 3:
        return times
    changed = True
    while changed:
        changed = False
        new: list[float] = [times[0]]
        i = 1
        while i < len(times) - 1:
            if times[i] - new[-1] < min_s:
                changed = True
                i += 1
                continue
            new.append(times[i])
            i += 1
        new.append(times[-1])
        times = new
    return times


def _expand_long(times: list[float], max_s: float) -> list[float]:
    """Split spans longer than max_s into equal-width sub-spans."""
    result: list[float] = [times[0]]
    for s, e in _pairs(times):
        span = e - s
        if span > max_s:
            n    = math.ceil(span / max_s)
            step = span / n
            for k in range(1, n):
                result.append(s + k * step)
        result.append(e)
    return sorted(set(round(t, 3) for t in result))


def _tier(score: int) -> str:
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MED_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _normalise_scores(events: list[SemanticEvent]) -> list[SemanticEvent]:
    """Replace importance=0 placeholders with normalised [0,1] values."""
    if not events:
        return events
    max_score = max(e.score for e in events) or 1
    return [e._replace(importance=round(e.score / max_score, 3)) for e in events]


# ── Public API ────────────────────────────────────────────────────────────────

def build_events(
    duration:            float,
    scene_cuts:          list[float] | None = None,
    transcript_segments: list[dict]  | None = None,
    extra_signals:       list[tuple[float, str]] | None = None,
    signal_stack:        list[BaseSignal] | None = None,
    max_event_s:         float = 36.0,
    min_event_s:         float = 5.0,
) -> list[SemanticEvent]:
    """
    Build a list of ``SemanticEvent`` objects covering [0, duration].

    Parameters
    ----------
    duration:            Total video length in seconds.
    scene_cuts:          Sorted cut timestamps from detect_scene_cuts().
                         Pass None to fall back to a single span.
    transcript_segments: Whisper segments list[{start, end, text}].
    extra_signals:       Caller-supplied (time, kind) pairs e.g. audio energy.
    signal_stack:        Override the default signal stack (useful for testing).
    max_event_s:         Hard cap on event duration (must match VLM token budget).
    min_event_s:         Minimum event duration; shorter spans are merged.

    Returns
    -------
    list[SemanticEvent] sorted by start, covering [0, duration] without gaps.
    """
    cuts     = sorted(scene_cuts or [0.0, duration])
    segs     = transcript_segments or []
    stack    = signal_stack if signal_stack is not None else DEFAULT_SIGNAL_STACK

    # Build unified weight map from active stack
    weight_map: dict[str, int] = {sig.name: sig.weight for sig in stack}

    # Collect all (time, kind) raw signals
    raw: list[tuple[float, str]] = []
    for sig in stack:
        raw.extend(sig.extract(duration=duration, scene_cuts=cuts, segments=segs))
    if extra_signals:
        raw.extend(extra_signals)

    # Fuse into scored boundaries
    boundaries = _fuse_signals(raw, weight_map)

    # Keep only boundaries above merge threshold
    strong = [b for b in boundaries if b.score >= MERGE_THRESHOLD]
    strong_lookup: dict[float, _Boundary] = {b.time: b for b in strong}

    # Build sorted unique boundary times anchored at 0.0 and duration
    times = sorted({0.0, duration} | {b.time for b in strong})

    # Collapse short spans
    times = _collapse_short(times, min_event_s)

    # Split long spans
    times = _expand_long(times, max_event_s)

    # Build SemanticEvent list (importance placeholder = 0, normalised below)
    events: list[SemanticEvent] = []
    for idx, (s, e) in enumerate(_pairs(times)):
        # Highest-scored boundary within this span
        span_boundaries = [b for b in strong if s <= b.time < e]
        span_score = max((b.score for b in span_boundaries), default=0)
        span_signals: dict[str, bool] = {}
        for b in span_boundaries:
            for k in b.signals:
                span_signals[k] = True

        t = _tier(span_score)
        events.append(SemanticEvent(
            event_id=idx,
            start=round(s, 3),
            end=round(e, 3),
            score=span_score,
            importance=0.0,   # filled by _normalise_scores
            profile=t,
            max_tokens=PROFILES[t].max_tokens,
            signals=span_signals,
        ))

    return _normalise_scores(events)


def events_to_chunks(
    events:    list[SemanticEvent],
    duration:  float,
    overlap_s: float = 3.0,
) -> list["Chunk"]:
    """
    Map ``SemanticEvent`` list → ``chunk_dispatch.Chunk`` list.

    Overlap padding is applied on each side (clamped to [0, duration]).
    chunk_id == event_id for direct lookup.
    """
    from chunk_dispatch import Chunk  # local import to avoid circular dep
    chunks = []
    for evt in events:
        chunks.append(Chunk(
            chunk_id=evt.event_id,
            scene_id=0,
            part_idx=evt.event_id,
            strict_start=evt.start,
            strict_end=evt.end,
            pad_start=round(max(0.0, evt.start - overlap_s), 3),
            pad_end=round(min(duration, evt.end + overlap_s), 3),
        ))
    return chunks


def event_stats(events: list[SemanticEvent]) -> dict:
    """Summary stats for logging."""
    by_tier: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    total_tokens = 0
    for e in events:
        by_tier[e.profile] = by_tier.get(e.profile, 0) + 1
        total_tokens += e.max_tokens
    return {
        "total_events":   len(events),
        "by_tier":        by_tier,
        "total_tokens":   total_tokens,
        "avg_duration_s": round(
            sum(e.end - e.start for e in events) / max(len(events), 1), 1
        ),
        "token_savings_vs_fixed": (
            f"{100 * (1 - total_tokens / (len(events) * PROFILES['HIGH'].max_tokens)):.0f}%"
            if events else "0%"
        ),
    }


# ── CLI (standalone test) ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Test semantic event builder")
    ap.add_argument("--duration",    type=float, required=True)
    ap.add_argument("--scene-cuts",  type=str,   help="JSON array e.g. '[0,45.2,112.7]'")
    ap.add_argument("--transcript",  type=str,   help="Path to transcripts_<ts>.json")
    ap.add_argument("--video",       type=str,   help="Video label for transcript lookup")
    ap.add_argument("--max-event-s", type=float, default=36.0)
    ap.add_argument("--min-event-s", type=float, default=5.0)
    args = ap.parse_args()

    cuts: list[float] | None = json.loads(args.scene_cuts) if args.scene_cuts else None
    segs: list[dict]         = []
    if args.transcript and args.video:
        with open(args.transcript) as f:
            td = json.load(f)
        for v in td.get("videos", []):
            if v.get("video") == args.video:
                segs = v.get("segments", [])
                break

    evts  = build_events(
        duration=args.duration,
        scene_cuts=cuts,
        transcript_segments=segs,
        max_event_s=args.max_event_s,
        min_event_s=args.min_event_s,
    )
    stats = event_stats(evts)
    print(json.dumps({
        "stats":  stats,
        "events": [e._asdict() for e in evts],
    }, indent=2))
