"""
Runs all three Experiment C0 weight configurations sequentially in one process.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_weight_ablation import WeightAblationExperiment  # noqa: E402


def main():
    config_path = str(Path(__file__).parent.parent.parent / "configs" / "brats.yaml")
    exp_dir = Path(__file__).parent
    epochs = 10
    seed = 0

    configs = [
        (0.8, 0.2),
        (0.5, 0.5),
        (0.2, 0.8),
    ]

    results = {}
    for focal_w, evid_w in configs:
        trainer = WeightAblationExperiment(
            config_path, exp_dir, seed=seed, focal_weight=focal_w, evidential_weight=evid_w
        )
        best_dice = trainer.train(epochs)
        results[trainer.tag] = best_dice

    print("\n" + "=" * 70)
    print("EXPERIMENT C0: ALL RUNS COMPLETE")
    print("=" * 70)
    for tag, dice in results.items():
        print(f"  {tag}: best_dice={dice:.4f}")


if __name__ == "__main__":
    main()
