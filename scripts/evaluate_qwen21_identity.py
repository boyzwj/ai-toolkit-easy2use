"""Compare fixed-prompt checkpoint samples with a person's reference gallery.

Optional dependencies: insightface and onnxruntime (or onnxruntime-gpu).
FaceAnalysis uses its local model cache; first use may download buffalo_l.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolkit.qwen_image21.evaluation import identity_scores


def image_paths(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--face-model", default="buffalo_l")
    args = parser.parse_args()
    try:
        import cv2
        from insightface.app import FaceAnalysis
    except ImportError as error:
        parser.error(f"Optional identity evaluation dependencies missing: {error}. Install insightface and onnxruntime.")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"])
    detector = FaceAnalysis(name=args.face_model, allowed_modules=["detection", "recognition"], providers=providers)
    detector.prepare(ctx_id=0 if args.device == "cuda" else -1, det_size=(640, 640))

    def embed(path):
        image = cv2.imread(str(path))
        if image is None:
            return None
        faces = detector.get(image)
        if not faces:
            return None
        # Match the subject in a portrait, and keep the rule the same for the
        # references and all checkpoint samples.
        largest = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        return largest.normed_embedding

    references, missing = [], []
    for path in image_paths(args.references):
        embedding = embed(path)
        if embedding is None:
            missing.append(str(path))
        else:
            references.append(embedding)
    if not references:
        parser.error("No faces detected in reference images")
    paths = image_paths(args.samples)
    if not paths:
        parser.error("No sample images found")
    report = identity_scores(references, [(path, embed(path)) for path in paths])
    report.update({"face_model": args.face_model, "requested_providers": providers, "references_without_face": missing})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "references_without_face")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
