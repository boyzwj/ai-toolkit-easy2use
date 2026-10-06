"""Identity evaluation independent of the training loss and optional face SDK."""

import numpy as np


def unit_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("Invalid face embedding")
    return vector / norm


def identity_scores(reference_embeddings, samples):
    """Cosine to the reference centroid, plus best-gallery match for context.

    Missing faces remain in the output instead of inflating an average by
    silently dropping hard samples. The SDK's embeddings are never saved.
    """
    if not reference_embeddings:
        raise ValueError("No reference face embeddings")
    gallery = np.stack([unit_vector(v) for v in reference_embeddings])
    centroid = unit_vector(gallery.mean(axis=0))
    rows = []
    for path, embedding in samples:
        if embedding is None:
            rows.append({"path": str(path), "face_detected": False, "cosine": None})
        else:
            vector = unit_vector(embedding)
            rows.append({"path": str(path), "face_detected": True,
                         "cosine": float(np.clip(vector @ centroid, -1, 1)),
                         "best_gallery_cosine": float(np.clip((gallery @ vector).max(), -1, 1))})
    valid = [row["cosine"] for row in rows if row["face_detected"]]
    return {"reference_faces": len(gallery), "samples": rows, "samples_total": len(rows),
            "faces_detected": len(valid), "mean_cosine": float(np.mean(valid)) if valid else None}
