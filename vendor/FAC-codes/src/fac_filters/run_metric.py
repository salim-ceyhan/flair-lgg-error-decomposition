import argparse
import csv
from pathlib import Path
from datetime import datetime
from time import perf_counter

from .analysis_utils import save_histogram_analysis
from .flow_utils import default_output_path, load_grayscale_image, resolve_input_path, save_result_image
from .INormalizedMironFlow import improved_normalized_miron_flow
from .IngardenFlow import ingarden_flow
from .NewMetric import NewMetric as NEW_METRIC_FN
from .NMFlow import normalized_miron_flow
from .RandersFlowV import randers_flow_v
from .beltrami2D import beltrami_2d


METRIC_DEFAULTS = {
    'newmetric': {'beta': 1.0, 'dt': 0.05, 'iterations': 3},
    'nm': {'beta': 0.2, 'dt': 0.2, 'iterations': 5},
    'inm': {'beta': 0.2, 'dt': 0.2, 'iterations': 5},
    'ingarden': {'beta': 0.2, 'dt': 0.1, 'iterations': 2},
    'randers': {'beta': 0.2, 'dt': 0.2, 'iterations': 2},
    'beltrami': {'beta': None, 'dt': 0.25, 'iterations': 15},
}

DEFAULT_EXTENSIONS = '.png,.jpg,.jpeg,.bmp,.tif,.tiff'


def run_metric(metric: str, image, beta: float | None, dt: float, iterations: int):
    if metric == 'newmetric':
        return NEW_METRIC_FN(image, beta, dt, iterations)
    if metric == 'nm':
        return normalized_miron_flow(image, beta, dt, iterations)
    if metric == 'inm':
        return improved_normalized_miron_flow(image, beta, dt, iterations)
    if metric == 'ingarden':
        return ingarden_flow(image, beta, dt, iterations)
    if metric == 'randers':
        return randers_flow_v(image, beta, dt, iterations)
    if metric == 'beltrami':
        return beltrami_2d(image, iterations, dt)
    raise ValueError(f'Unknown metric: {metric}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run any available metric flow from a single entry point.')
    parser.add_argument(
        '--metric',
        required=True,
        choices=sorted(METRIC_DEFAULTS.keys()),
        help='Metric/flow to apply.',
    )
    parser.add_argument('--image', type=Path, help='Path to the input image. If omitted, a file picker opens.')
    parser.add_argument('--input-dir', type=Path, help='Process all supported images in this directory.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively scan subdirectories when using --input-dir.',
    )
    parser.add_argument(
        '--extensions',
        default=DEFAULT_EXTENSIONS,
        help='Comma-separated list of image extensions for batch mode.',
    )
    parser.add_argument('--beta', type=float, default=None, help='Aspect ratio parameter. Ignored by beltrami.')
    parser.add_argument('--dt', type=float, default=None, help='Time step.')
    parser.add_argument('--iterations', type=int, default=None, help='Number of iterations.')
    parser.add_argument('--output', type=Path, help='Output image path.')
    parser.add_argument('--output-dir', type=Path, help='Directory for batch outputs.')
    parser.add_argument(
        '--output-subdir',
        default='Filtered',
        help='Default output subdirectory name when --output/--output-dir is not given.',
    )
    parser.add_argument(
        '--timestamp-subdir',
        action='store_true',
        help='Append a timestamp to the default output subdirectory name.',
    )
    parser.add_argument(
        '--save-analysis',
        action='store_true',
        help='Save a visual analysis report with original, filtered, difference image, and histogram comparison.',
    )
    parser.add_argument(
        '--analysis-subdir',
        default='Analysis',
        help='Subdirectory name for analysis images when --save-analysis is used.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip batch outputs that already exist.',
    )
    parser.add_argument(
        '--report-csv',
        type=Path,
        help='Write a CSV report for batch processing.',
    )
    parser.add_argument('--no-show', action='store_true', help='Save the result without opening it.')
    return parser.parse_args()


def parse_extensions(extensions_arg: str) -> set[str]:
    extensions = {
        ext.strip().lower() if ext.strip().startswith('.') else f".{ext.strip().lower()}"
        for ext in extensions_arg.split(',')
        if ext.strip()
    }
    if not extensions:
        raise ValueError('At least one extension must be provided.')
    return extensions


def collect_image_paths(input_dir: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Input path is not a directory: {input_dir}')

    iterator = input_dir.rglob('*') if recursive else input_dir.iterdir()
    image_paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in extensions)
    if not image_paths:
        raise FileNotFoundError(f'No supported image files found in: {input_dir}')
    return image_paths


def process_one(
    metric: str,
    image_path: Path,
    beta: float | None,
    dt: float,
    iterations: int,
    output_path: Path,
    show: bool,
    save_analysis: bool,
    analysis_path: Path | None,
) -> float:
    image = load_grayscale_image(image_path)
    assert image is not None, f'image not found: {image_path}'

    start = perf_counter()
    result = run_metric(metric, image, beta, dt, iterations)
    elapsed = perf_counter() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_result_image(result, output_path, show=show)
    if save_analysis and analysis_path is not None:
        save_histogram_analysis(image, result, analysis_path)

    print(f'Metric: {metric}')
    print(f'Input: {image_path}')
    print(f'Output: {output_path}')
    if save_analysis and analysis_path is not None:
        print(f'Analysis: {analysis_path}')
    if beta is not None:
        print(f'beta={beta}, dt={dt}, iterations={iterations}')
    else:
        print(f'dt={dt}, iterations={iterations}')
    print(f'Elapsed time: {elapsed:.3f}s')
    return elapsed


def main() -> None:
    args = parse_args()
    defaults = METRIC_DEFAULTS[args.metric]
    extensions = parse_extensions(args.extensions)
    output_subdir = args.output_subdir
    if args.timestamp_subdir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subdir = f'{output_subdir}_{timestamp}'

    beta = defaults['beta'] if args.beta is None else args.beta
    dt = defaults['dt'] if args.dt is None else args.dt
    iterations = defaults['iterations'] if args.iterations is None else args.iterations

    if args.image is not None and args.input_dir is not None:
        raise SystemExit('Use either --image or --input-dir, not both.')

    if args.input_dir is not None:
        output_dir = args.output_dir if args.output_dir is not None else args.input_dir / output_subdir
        image_paths = collect_image_paths(args.input_dir, args.recursive, extensions)
        output_dir_resolved = output_dir.resolve()
        image_paths = [
            path for path in image_paths
            if output_dir_resolved not in path.resolve().parents and path.resolve() != output_dir_resolved
        ]
        if not image_paths:
            raise FileNotFoundError('No input images remain after excluding the output directory.')
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Processing {len(image_paths)} image(s) from: {args.input_dir}')
        report_rows: list[dict[str, str]] = []
        for image_path in image_paths:
            relative_parent = image_path.parent.relative_to(args.input_dir)
            output_path = output_dir / relative_parent / f'{image_path.stem}_{args.metric}.png'
            analysis_path = None
            if args.save_analysis:
                analysis_path = output_dir / relative_parent / args.analysis_subdir / f'{image_path.stem}_{args.metric}_analysis.png'
            if args.skip_existing and output_path.exists():
                print(f'Skipped existing: {output_path}')
                report_rows.append(
                    {
                        'metric': args.metric,
                        'input': str(image_path),
                        'output': str(output_path),
                        'analysis': '' if analysis_path is None else str(analysis_path),
                        'status': 'skipped_existing',
                        'elapsed_seconds': '',
                        'beta': '' if beta is None else str(beta),
                        'dt': str(dt),
                        'iterations': str(iterations),
                    }
                )
                continue

            elapsed = process_one(
                args.metric,
                image_path,
                beta,
                dt,
                iterations,
                output_path,
                show=False,
                save_analysis=args.save_analysis,
                analysis_path=analysis_path,
            )
            report_rows.append(
                {
                    'metric': args.metric,
                    'input': str(image_path),
                    'output': str(output_path),
                    'analysis': '' if analysis_path is None else str(analysis_path),
                    'status': 'processed',
                    'elapsed_seconds': f'{elapsed:.6f}',
                    'beta': '' if beta is None else str(beta),
                    'dt': str(dt),
                    'iterations': str(iterations),
                }
            )

        if args.report_csv is not None:
            args.report_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.report_csv.open('w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        'metric',
                        'input',
                        'output',
                        'analysis',
                        'status',
                        'elapsed_seconds',
                        'beta',
                        'dt',
                        'iterations',
                    ],
                )
                writer.writeheader()
                writer.writerows(report_rows)
            print(f'CSV report saved to: {args.report_csv}')
        return

    image_path = resolve_input_path(args.image)
    output_path = (
        args.output
        if args.output is not None
        else default_output_path(image_path, args.metric, output_subdir=output_subdir)
    )
    analysis_path = None
    if args.save_analysis:
        analysis_path = output_path.parent / args.analysis_subdir / f'{output_path.stem}_analysis.png'
    process_one(
        args.metric,
        image_path,
        beta,
        dt,
        iterations,
        output_path,
        show=not args.no_show,
        save_analysis=args.save_analysis,
        analysis_path=analysis_path,
    )


if __name__ == '__main__':
    main()
