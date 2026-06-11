#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image, ImageDraw


CIFAR_REPO = Path("/ssd1/hsiao/statpruning/pytorch_resnet_cifar10-master")
if str(CIFAR_REPO) not in sys.path:
    sys.path.insert(0, str(CIFAR_REPO))

# The shared model file imports thop only for an optional FLOPs test under
# __main__. Evaluation does not need it, so avoid adding a dependency.
if "thop" not in sys.modules:
    thop_stub = types.ModuleType("thop")
    thop_stub.profile = lambda *args, **kwargs: (0, 0)
    sys.modules["thop"] = thop_stub

import resnet_cifar10  # noqa: E402


CORRUPTIONS = [
    "gaussian_noise",
    "motion_blur",
    "fog",
    "brightness",
    "jpeg_compression",
]
SEVERITIES = [1, 3, 5]
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR-10 pretrained ResNets on CIFAR-10-C."
    )
    parser.add_argument("--cifar10_root", default="/ssd1/datasets/cifar10")
    parser.add_argument("--cifar10c_root", default="/ssd3/fan/hw/data/CIFAR-10-C")
    parser.add_argument("--checkpoint_root", default=str(CIFAR_REPO / "pretrained_models"))
    parser.add_argument("--out_dir", default="/ssd3/fan/hw/cifar10c_audit/outputs")
    parser.add_argument("--models", nargs="+", default=["resnet20", "resnet56"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_model(name, checkpoint_root, device):
    if not hasattr(resnet_cifar10, name):
        raise ValueError(f"Unknown model architecture: {name}")
    model = getattr(resnet_cifar10, name)()
    ckpt_path = Path(checkpoint_root) / f"{name}.th"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_uint8(batch, device):
    tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    return (tensor - MEAN.to(device)) / STD.to(device)


def evaluate_array(model, images, labels, indices, batch_size, device):
    preds = []
    confs = []
    total_correct = 0
    wrong_conf_sum = 0.0
    wrong_count = 0

    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            batch = preprocess_uint8(images[idx], device)
            target = torch.from_numpy(np.asarray(labels[idx], dtype=np.int64)).to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
            correct = pred.eq(target)
            total_correct += int(correct.sum().item())
            wrong = ~correct
            if wrong.any():
                wrong_conf_sum += float(conf[wrong].sum().item())
                wrong_count += int(wrong.sum().item())
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())

    pred_np = np.concatenate(preds)
    conf_np = np.concatenate(confs)
    acc = total_correct / len(indices)
    wrong_conf = wrong_conf_sum / wrong_count if wrong_count else 0.0
    return {
        "accuracy": acc,
        "wrong_confidence": wrong_conf,
        "pred": pred_np,
        "confidence": conf_np,
        "correct": pred_np == np.asarray(labels[indices]),
    }


def load_clean_test(root):
    ds = datasets.CIFAR10(root=root, train=False, download=False)
    images = np.asarray(ds.data, dtype=np.uint8)
    labels = np.asarray(ds.targets, dtype=np.int64)
    return images, labels


def make_indices(sample_size, seed):
    if sample_size <= 0 or sample_size >= 10000:
        return np.arange(10000, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(10000, size=sample_size, replace=False)).astype(np.int64)


def condition_slice(severity):
    start = (severity - 1) * 10000
    end = severity * 10000
    return start, end


def write_results(rows, out_dir):
    path = out_dir / "results.csv"
    fieldnames = [
        "model",
        "corruption",
        "severity",
        "n_samples",
        "clean_acc",
        "shifted_acc",
        "accuracy_drop",
        "relative_drop",
        "wrong_confidence",
        "failure_overlap",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def plot_heatmaps(rows, out_dir, models):
    for model in models:
        subset = [r for r in rows if r["model"] == model and r["severity"] != 0]
        acc = np.zeros((len(SEVERITIES), len(CORRUPTIONS)), dtype=float)
        drop = np.zeros_like(acc)
        for i, severity in enumerate(SEVERITIES):
            for j, corruption in enumerate(CORRUPTIONS):
                row = next(
                    r
                    for r in subset
                    if r["severity"] == severity and r["corruption"] == corruption
                )
                acc[i, j] = row["shifted_acc"]
                drop[i, j] = row["relative_drop"]

        for values, title, filename in [
            (acc, "Accuracy", f"{model}_accuracy_heatmap.png"),
            (drop, "Relative Drop", f"{model}_relative_drop_heatmap.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 3.8))
            im = ax.imshow(values, aspect="auto", cmap="viridis")
            ax.set_title(f"{model} {title}")
            ax.set_xticks(range(len(CORRUPTIONS)), CORRUPTIONS, rotation=35, ha="right")
            ax.set_yticks(range(len(SEVERITIES)), [f"severity {s}" for s in SEVERITIES])
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=180)
            plt.close(fig)


def verdict(ok_count, total_count):
    if ok_count == total_count:
        return "Supported"
    if ok_count == 0:
        return "Refuted"
    return "Partially supported"


def write_claim_audit(rows, out_dir, models):
    lookup = {
        (r["model"], r["corruption"], r["severity"]): r
        for r in rows
        if r["severity"] != 0
    }
    audits = []

    checks = []
    for model in models:
        for severity in SEVERITIES:
            g = lookup[(model, "gaussian_noise", severity)]["relative_drop"]
            b = lookup[(model, "brightness", severity)]["relative_drop"]
            checks.append(g > b)
    audits.append(
        {
            "claim_id": "C1",
            "claim": "Gaussian noise is more damaging than brightness.",
            "testable_hypothesis": "For the same model and severity, Gaussian noise has larger relative accuracy drop than brightness.",
            "verdict": verdict(sum(checks), len(checks)),
            "evidence": f"{sum(checks)}/{len(checks)} model-severity comparisons satisfy gaussian_noise > brightness.",
        }
    )

    checks = []
    for model in models:
        for corruption in CORRUPTIONS:
            accs = [lookup[(model, corruption, s)]["shifted_acc"] for s in SEVERITIES]
            checks.append(accs[0] >= accs[1] >= accs[2])
    audits.append(
        {
            "claim_id": "C2",
            "claim": "Higher corruption severity lowers model accuracy.",
            "testable_hypothesis": "For each model and corruption, severity 1 -> 3 -> 5 accuracy is monotonically non-increasing.",
            "verdict": verdict(sum(checks), len(checks)),
            "evidence": f"{sum(checks)}/{len(checks)} model-corruption curves are monotonic.",
        }
    )

    checks = []
    for model in models:
        jpeg = np.mean([lookup[(model, "jpeg_compression", s)]["relative_drop"] for s in SEVERITIES])
        noise = np.mean([lookup[(model, "gaussian_noise", s)]["relative_drop"] for s in SEVERITIES])
        checks.append(jpeg < noise)
    audits.append(
        {
            "claim_id": "C3",
            "claim": "JPEG compression is easier than Gaussian noise.",
            "testable_hypothesis": "Average relative drop for JPEG compression is lower than Gaussian noise.",
            "verdict": verdict(sum(checks), len(checks)),
            "evidence": f"{sum(checks)}/{len(checks)} models satisfy JPEG average drop < Gaussian noise average drop.",
        }
    )

    model_a, model_b = models[0], models[1]
    avg_a = np.mean(
        [lookup[(model_a, c, s)]["relative_drop"] for c in CORRUPTIONS for s in SEVERITIES]
    )
    avg_b = np.mean(
        [lookup[(model_b, c, s)]["relative_drop"] for c in CORRUPTIONS for s in SEVERITIES]
    )
    audits.append(
        {
            "claim_id": "C4",
            "claim": "The stronger model has a smaller relative drop under corruptions.",
            "testable_hypothesis": f"{model_b} average relative drop is lower than {model_a}.",
            "verdict": "Supported" if avg_b < avg_a else "Refuted",
            "evidence": f"{model_a} avg relative drop={avg_a:.4f}; {model_b} avg relative drop={avg_b:.4f}.",
        }
    )

    sev1 = np.mean(
        [lookup[(models[0], c, 1)]["failure_overlap"] for c in CORRUPTIONS]
    )
    sev5 = np.mean(
        [lookup[(models[0], c, 5)]["failure_overlap"] for c in CORRUPTIONS]
    )
    audits.append(
        {
            "claim_id": "C5",
            "claim": "At high severity, models fail on the same hard samples more often.",
            "testable_hypothesis": "Average failure overlap at severity 5 is higher than severity 1.",
            "verdict": "Supported" if sev5 > sev1 else "Refuted",
            "evidence": f"Mean failure overlap severity1={sev1:.4f}; severity5={sev5:.4f}.",
        }
    )

    path = out_dir / "claim_audit.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["claim_id", "claim", "testable_hypothesis", "verdict", "evidence"],
        )
        writer.writeheader()
        writer.writerows(audits)
    return path


def save_case_image(image, title, path):
    img = Image.fromarray(image)
    canvas = Image.new("RGB", (img.width, img.height + 18), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((1, img.height + 2), title[:80], fill=(0, 0, 0))
    canvas.save(path)


def choose_failure_cases(pred_store, out_dir, indices):
    cases = []
    case_dir = out_dir / "failure_cases"
    case_dir.mkdir(exist_ok=True)

    def add_case(case_type, model_name, condition, local_pos, image, labels, pred, conf):
        if len(cases) >= 5:
            return
        global_index = int(indices[local_pos])
        true_label = int(labels[local_pos])
        pred_label = int(pred[local_pos])
        filename = f"case_{len(cases) + 1}_{case_type}.png"
        save_case_image(
            image[local_pos],
            f"{case_type}: {CIFAR10_CLASSES[true_label]} -> {CIFAR10_CLASSES[pred_label]} ({conf[local_pos]:.2f})",
            case_dir / filename,
        )
        cases.append(
            {
                "case_id": f"case_{len(cases) + 1}",
                "case_type": case_type,
                "model": model_name,
                "corruption": condition[0],
                "severity": condition[1],
                "sample_index": global_index,
                "true_label": CIFAR10_CLASSES[true_label],
                "prediction": CIFAR10_CLASSES[pred_label],
                "confidence": f"{float(conf[local_pos]):.6f}",
                "image_path": str(case_dir / filename),
                "explanation": "Representative failure selected from the evaluated condition.",
            }
        )

    for condition in [("gaussian_noise", 5), ("motion_blur", 5), ("fog", 5), ("jpeg_compression", 5), ("brightness", 1)]:
        entries = pred_store[condition]
        model_names = list(entries.keys())
        if len(model_names) < 2:
            continue
        a, b = model_names[0], model_names[1]
        labels = entries[a]["labels"]
        image = entries[a]["images"]
        wrong_a = ~entries[a]["correct"]
        wrong_b = ~entries[b]["correct"]

        candidates = [
            ("both_wrong", a, wrong_a & wrong_b, entries[a]),
            (f"{a}_wrong_{b}_correct", a, wrong_a & ~wrong_b, entries[a]),
            (f"{b}_wrong_{a}_correct", b, wrong_b & ~wrong_a, entries[b]),
        ]
        for case_type, model_name, mask, entry in candidates:
            if len(cases) >= 5:
                break
            if mask.any():
                masked = np.where(mask)[0]
                best = masked[np.argmax(entry["confidence"][masked])]
                add_case(
                    case_type,
                    model_name,
                    condition,
                    int(best),
                    image,
                    labels,
                    entry["pred"],
                    entry["confidence"],
                )
        if len(cases) >= 5:
            break

    path = out_dir / "failure_cases.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "case_type",
                "model",
                "corruption",
                "severity",
                "sample_index",
                "true_label",
                "prediction",
                "confidence",
                "image_path",
                "explanation",
            ],
        )
        writer.writeheader()
        writer.writerows(cases)
    return path


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.smoke:
        out_dir = out_dir / "smoke"
        sample_size = args.sample_size or 128
    else:
        out_dir = out_dir / "full"
        sample_size = args.sample_size
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = make_indices(sample_size, args.seed)
    clean_images, clean_labels = load_clean_test(args.cifar10_root)
    c_labels = np.load(Path(args.cifar10c_root) / "labels.npy", mmap_mode="r")

    models = {
        name: load_model(name, args.checkpoint_root, device)
        for name in args.models
    }
    rows = []
    clean_results = {}
    pred_store = {}

    for name, model in models.items():
        clean = evaluate_array(model, clean_images, clean_labels, indices, args.batch_size, device)
        clean_results[name] = clean
        rows.append(
            {
                "model": name,
                "corruption": "clean",
                "severity": 0,
                "n_samples": len(indices),
                "clean_acc": clean["accuracy"],
                "shifted_acc": clean["accuracy"],
                "accuracy_drop": 0.0,
                "relative_drop": 0.0,
                "wrong_confidence": clean["wrong_confidence"],
                "failure_overlap": "",
            }
        )

    for corruption in CORRUPTIONS:
        corruption_images = np.load(Path(args.cifar10c_root) / f"{corruption}.npy", mmap_mode="r")
        for severity in SEVERITIES:
            start, _ = condition_slice(severity)
            condition_indices = indices + start
            condition = (corruption, severity)
            pred_store[condition] = {}
            for name, model in models.items():
                shifted = evaluate_array(
                    model,
                    corruption_images,
                    c_labels,
                    condition_indices,
                    args.batch_size,
                    device,
                )
                clean_acc = clean_results[name]["accuracy"]
                drop = clean_acc - shifted["accuracy"]
                rel_drop = drop / clean_acc if clean_acc else 0.0
                pred_store[condition][name] = {
                    **shifted,
                    "images": np.asarray(corruption_images[condition_indices]),
                    "labels": np.asarray(c_labels[condition_indices]),
                }
                rows.append(
                    {
                        "model": name,
                        "corruption": corruption,
                        "severity": severity,
                        "n_samples": len(indices),
                        "clean_acc": clean_acc,
                        "shifted_acc": shifted["accuracy"],
                        "accuracy_drop": drop,
                        "relative_drop": rel_drop,
                        "wrong_confidence": shifted["wrong_confidence"],
                        "failure_overlap": "",
                    }
                )

            model_names = list(models.keys())
            wrong_a = ~pred_store[condition][model_names[0]]["correct"]
            wrong_b = ~pred_store[condition][model_names[1]]["correct"]
            overlap = float(np.mean(wrong_a & wrong_b))
            for row in rows:
                if row["corruption"] == corruption and row["severity"] == severity:
                    row["failure_overlap"] = overlap

    results_path = write_results(rows, out_dir)
    claim_path = write_claim_audit(rows, out_dir, list(models.keys()))
    failure_path = choose_failure_cases(pred_store, out_dir, indices)
    plot_heatmaps(rows, out_dir, list(models.keys()))

    metadata = {
        "mode": "smoke" if args.smoke else "full",
        "device": str(device),
        "models": list(models.keys()),
        "corruptions": CORRUPTIONS,
        "severities": SEVERITIES,
        "sample_size": int(len(indices)),
        "results_csv": str(results_path),
        "claim_audit_csv": str(claim_path),
        "failure_cases_csv": str(failure_path),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
