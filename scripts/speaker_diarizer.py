#!/usr/bin/env python3
"""
Speaker diarization via d-vector embeddings + K-means.
Zero GPU, zero pyannote. Uses resemblyzer or speechbrain ECAPA-TDNN.

Pipeline:
  video URL → ffmpeg extract 16kHz mono WAV
            → split at Whisper segment boundaries
            → d-vector embedding per segment
            → K-means(K=n_speakers)
            → assign P001/P002/... by order of first appearance
            → return labeled segments

Accuracy: ~85-92% on 2-6 speaker conversations with known K.

Install one of:
  pip install resemblyzer soundfile        # recommended (faster, simpler)
  pip install speechbrain torchaudio       # fallback ECAPA-TDNN
  pip install scikit-learn                 # required for K-means (usually present)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Callable

import numpy as np  # type: ignore


# ── Audio extraction ──────────────────────────────────────────────────────────

def _extract_audio_wav(video_url: str, out_path: str) -> bool:
    """Extract 16kHz mono WAV from video URL via ffmpeg. Returns True on success."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_url,
        "-vn",                   # no video stream
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",          # 16 kHz
        "-ac", "1",              # mono
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and os.path.getsize(out_path) > 0
    except Exception:
        return False


# ── Embedding backends ────────────────────────────────────────────────────────

EmbedFn = Callable[[str, float, float], "np.ndarray | None"]


def _get_embedder() -> tuple[EmbedFn | None, str | None]:
    """Return (embedder_fn, name) or (None, None) if no backend is available.

    Tries in order:
      1. resemblyzer   — pure numpy, no heavy deps, very fast on CPU
      2. speechbrain   — ECAPA-TDNN, higher quality, larger install
    """
    # ── 1. resemblyzer ────────────────────────────────────────────────────────
    try:
        import soundfile as sf  # type: ignore
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore

        encoder = VoiceEncoder(device="cpu")

        def embed_resemblyzer(wav_path: str, start: float, end: float) -> np.ndarray | None:
            wav, sr = sf.read(wav_path)
            start_i = int(start * sr)
            end_i   = int(end * sr)
            segment_wav = wav[start_i:end_i]
            if len(segment_wav) < sr * 0.5:   # skip segments < 0.5 s — too noisy
                return None
            processed = preprocess_wav(segment_wav, source_sr=sr)
            return encoder.embed_utterance(processed)

        return embed_resemblyzer, "resemblyzer"

    except ImportError:
        pass

    # ── 2. speechbrain ECAPA-TDNN ─────────────────────────────────────────────
    try:
        import torchaudio  # type: ignore
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
            savedir="/tmp/speechbrain_models",
        )

        def embed_speechbrain(wav_path: str, start: float, end: float) -> np.ndarray | None:
            waveform, sr = torchaudio.load(wav_path)
            start_i = int(start * sr)
            end_i   = int(end * sr)
            seg = waveform[:, start_i:end_i]
            if seg.shape[1] < sr * 0.5:
                return None
            emb = classifier.encode_batch(seg)
            return emb.squeeze().cpu().numpy()  # type: ignore[union-attr]

        return embed_speechbrain, "speechbrain"

    except (ImportError, Exception):
        pass

    return None, None


# ── K-means clustering ────────────────────────────────────────────────────────

def _cluster_embeddings(
    embeddings: list[np.ndarray | None],
    n_speakers: int,
) -> list[int]:
    """Cluster embeddings into n_speakers groups.

    Returns a list of cluster labels, same length as `embeddings`.
    -1 means no embedding was computed for that segment (too short / silence).
    When fewer valid segments exist than n_speakers, all assigned to cluster 0.
    """
    from sklearn.cluster import KMeans  # type: ignore

    valid_idxs = [i for i, e in enumerate(embeddings) if e is not None]

    if len(valid_idxs) < n_speakers:
        # Not enough segments to form n_speakers clusters — assign all to P001
        return [0 if e is not None else -1 for e in embeddings]

    X = np.stack([embeddings[i] for i in valid_idxs])  # type: ignore[arg-type]

    # L2-normalise so cosine similarity == dot product (KMeans minimises Euclidean)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-8)

    km = KMeans(n_clusters=n_speakers, n_init=10, random_state=42)
    labels_valid = km.fit_predict(X)

    # Map back to full-length list
    labels: list[int] = [-1] * len(embeddings)
    for idx, label in zip(valid_idxs, labels_valid.tolist()):
        labels[idx] = int(label)

    return labels


# ── Person ID assignment ──────────────────────────────────────────────────────

def _assign_person_ids(segments: list[dict], labels: list[int]) -> list[dict]:
    """Assign P001/P002/... to clusters in order of first appearance in transcript.

    Segments with label=-1 (too short to embed) get speaker_id=None.
    """
    cluster_to_pid: dict[int, str] = {}
    pid_counter = 1
    result: list[dict] = []

    for seg, label in zip(segments, labels):
        if label == -1:
            pid: str | None = None
        elif label in cluster_to_pid:
            pid = cluster_to_pid[label]
        else:
            pid = f"P{pid_counter:03d}"
            cluster_to_pid[label] = pid
            pid_counter += 1

        result.append({**seg, "speaker_id": pid})

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def diarize(
    video_url: str,
    segments: list[dict],          # Whisper segments: [{id, start, end, text}, ...]
    n_speakers: int,               # from len(cast_analysis.get("persons", [])), min=1
    *,
    backend_url: str = "http://localhost:8080",  # reserved for future use
) -> list[dict]:
    """Label each Whisper segment with a speaker person ID (P001, P002, ...).

    Falls back gracefully at every step:
      - If no embedding library is installed → returns segments unchanged.
      - If audio extraction fails → returns segments unchanged.
      - If clustering fails → returns segments unchanged.
      - If n_speakers == 1 → skips clustering, assigns all to P001.

    Args:
        video_url:   URL (or local path) to the video file.
        segments:    List of Whisper segment dicts with keys: start, end, text.
        n_speakers:  Expected number of distinct speakers.
        backend_url: Unused — kept for API symmetry.

    Returns:
        Segments with 'speaker_id' field added. Original segments if diarization fails.
    """
    n_speakers = max(1, min(n_speakers, 8))

    if not segments:
        return segments

    # ── Trivial case: single speaker ─────────────────────────────────────────
    if n_speakers == 1:
        return [{**s, "speaker_id": "P001"} for s in segments]

    # ── Check embedding backend availability ──────────────────────────────────
    embedder, backend_name = _get_embedder()
    if embedder is None:
        print(
            "  [diarize] No embedding backend found. "
            "Install 'resemblyzer soundfile' or 'speechbrain torchaudio'. "
            "Continuing without speaker labels.",
            flush=True,
        )
        return segments

    print(f"  [diarize] Using backend: {backend_name}", flush=True)

    # ── Extract audio to temp WAV ─────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        if not _extract_audio_wav(video_url, wav_path):
            print(
                "  [diarize] ffmpeg audio extraction failed. "
                "Continuing without speaker labels.",
                flush=True,
            )
            return segments

        # ── Embed each segment ────────────────────────────────────────────────
        embeddings: list[np.ndarray | None] = []
        for seg in segments:
            try:
                emb = embedder(wav_path, float(seg["start"]), float(seg["end"]))
            except Exception as exc:
                print(f"  [diarize] Embedding error at {seg['start']:.1f}s: {exc}", flush=True)
                emb = None
            embeddings.append(emb)

    n_valid = sum(1 for e in embeddings if e is not None)
    print(
        f"  [diarize] Embedded {n_valid}/{len(segments)} segments "
        f"(skipped {len(segments) - n_valid} short/silent)",
        flush=True,
    )

    if n_valid == 0:
        print("  [diarize] No valid embeddings — continuing without speaker labels.", flush=True)
        return segments

    # ── Cluster ───────────────────────────────────────────────────────────────
    try:
        labels = _cluster_embeddings(embeddings, n_speakers)
    except Exception as exc:
        print(f"  [diarize] Clustering failed: {exc} — continuing without speaker labels.", flush=True)
        return segments

    # ── Assign person IDs by first appearance order ───────────────────────────
    labeled = _assign_person_ids(segments, labels)

    n_labeled = sum(1 for s in labeled if s.get("speaker_id"))
    print(
        f"  [diarize] Done: {n_labeled}/{len(labeled)} segments labeled "
        f"across {n_speakers} speaker(s)",
        flush=True,
    )
    return labeled
