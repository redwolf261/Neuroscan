"""
Phase 8 experiment runner. Run from this directory:

    cd research_infra/phase8
    python run_experiment.py --mode baseline --epochs 25 --out-csv ../baseline_metrics.csv
    python run_experiment.py --mode sampling --epochs 25 --out-csv ../sampling_metrics.csv
    python run_experiment.py --mode optimization --epochs 25 --out-csv ../optimization_metrics.csv

Each mode reuses the identical underlying training/validation code
(final_model.py's train_segmentation_epoch / validate_segmentation) and the
identical MAE-pretrained encoder checkpoint; the only difference between
modes is the one controlled variable each introduces (see common.py).
"""
import argparse
import time

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'sampling', 'optimization'], required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--grad-scale-factor', type=float, default=1.0 / 3.0,
                         help="Only used for --mode optimization; see common.py's "
                              "make_gradient_scale_hook docstring for justification.")
    args = parser.parse_args()

    fm = common.fm
    device = fm.device
    print(f"[phase8] device={device}  cuda_available={__import__('torch').cuda.is_available()}")

    t_start = time.time()

    if args.mode == 'baseline':
        common.run_fixed_epoch_experiment(
            'baseline', args.epochs, device,
            selection_mode='adaptive', gradient_scale_hook=None,
            out_csv_path=args.out_csv,
        )
    elif args.mode == 'sampling':
        common.run_fixed_epoch_experiment(
            'sampling_prototype', args.epochs, device,
            selection_mode='uniform', gradient_scale_hook=None,
            out_csv_path=args.out_csv,
        )
    elif args.mode == 'optimization':
        hook = common.make_gradient_scale_hook(scale_factor=args.grad_scale_factor)
        common.run_fixed_epoch_experiment(
            'optimization_prototype', args.epochs, device,
            selection_mode='adaptive', gradient_scale_hook=hook,
            out_csv_path=args.out_csv,
        )

    print(f"\n[phase8] Total wall-clock time for this experiment: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
