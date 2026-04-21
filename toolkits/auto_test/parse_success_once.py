#!/usr/bin/env python3
import re
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean

# Import check_global_step function
from check import check_global_step


def parse_log_file(log_path: str) -> Tuple[str, Dict[int, float]]:
    """
    Parse log file to extract experiment name and success_once values for each step
    
    Args:
        log_path: Log file path
        
    Returns:
        experiment_name: Experiment name
        step_to_success: Mapping dictionary from step to success_once
    """
    experiment_name = ""
    step_to_success = {}
    
    # Extract experiment name from path
    # Path format: .../20260413-02:04:24-libero_90_grpo_openvlaoft/run_embodiment.log
    path_obj = Path(log_path)
    parent_dir = path_obj.parent.name
    if parent_dir:
        # Extract the part after date-time as experiment name
        match = re.match(r'\d{8}-\d{2}:\d{2}:\d{2}-(.+)', parent_dir)
        if match:
            experiment_name = match.group(1)
        else:
            experiment_name = parent_dir
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Use regex to match Metric Table blocks
    # Match Global Step: X/1000 and success_once=xxx
    metric_pattern = r'Global Step:\s*(\d+)/\d+.*?success_once=([\d.]+)'
    matches = re.findall(metric_pattern, content, re.DOTALL)
    
    for match in matches:
        step = int(match[0])
        success_once = float(match[1])
        step_to_success[step] = success_once
    
    return experiment_name, step_to_success


def get_target_success_once(step_to_success: Dict[int, float], target_step: int = 100) -> Tuple[int, Optional[float]]:
    """
    Get success_once value for target step, if not exists return the last step's value
    
    Args:
        step_to_success: Mapping dictionary from step to success_once
        target_step: Target step, default is 100
        
    Returns:
        actual_step: Actual step used
        success_once: Corresponding success_once value
    """
    if not step_to_success:
        return 0, None
    
    if target_step in step_to_success:
        return target_step, step_to_success[target_step]
    
    # If target step doesn't exist, return the last step
    last_step = max(step_to_success.keys())
    return last_step, step_to_success[last_step]


def process_single_log(log_path: str, target_step: int = 100, threshold: Optional[float] = None) -> Dict:
    """
    Process a single log file
    
    Args:
        log_path: Log file path
        target_step: Target step
        threshold: Threshold for check_global_step
        
    Returns:
        Dictionary containing results
    """
    experiment_name, step_to_success = parse_log_file(log_path)
    actual_step, success_once = get_target_success_once(step_to_success, target_step)
    
    # Call check_global_step
    reached_threshold, crashed_before_threshold = check_global_step(log_path, threshold, verbose=False)
    
    result = {
        'log_path': log_path,
        'experiment_name': experiment_name,
        'target_step': target_step,
        'actual_step': actual_step,
        'success_once': success_once,
        'total_steps_found': len(step_to_success),
        'max_step': max(step_to_success.keys()) if step_to_success else 0,
        'found_target': target_step in step_to_success,
        'reached_threshold': reached_threshold,
        'crashed_before_threshold': crashed_before_threshold,
        'step_to_success': step_to_success  # Save success_once data for each step
    }
    
    return result


def process_log_directory(log_dir: str, target_step: int = 100, log_filename: str = "run_embodiment.log", threshold: Optional[float] = None) -> list:
    """
    Process all log files in directory
    
    Args:
        log_dir: Log directory path
        target_step: Target step
        log_filename: Log filename
        threshold: Threshold for check_global_step
        
    Returns:
        List of results
    """
    results = []
    log_dir = Path(log_dir)
    
    # Find all matching log files
    for log_file in log_dir.rglob(log_filename):
        try:
            result = process_single_log(str(log_file), target_step, threshold)
            results.append(result)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    return results


def print_results(results: list):
    """
    Print results
    
    Args:
        results: List of results
    """
    print("=" * 140)
    print(f"{'Experiment Name':<45} {'Step':<8} {'success_once':<14} {'reached':<10} {'crashed':<10} {'Note':<30}")
    print("=" * 140)
    
    for r in results:
        note = "Target step found" if r['found_target'] else f"Using last step (total {r['total_steps_found']} steps)"
        success_str = f"{r['success_once']:.6f}" if r['success_once'] is not None else "N/A"
        reached_str = str(r['reached_threshold'])
        crashed_str = str(r['crashed_before_threshold'])
        print(f"{r['experiment_name']:<45} {r['actual_step']:<8} {success_str:<14} {reached_str:<10} {crashed_str:<10} {note:<30}")
    
    print("=" * 140)


def plot_success_once_curves(results: list, output_path: str = None, figsize: tuple = (12, 8)):
    """
    Plot success_once curves for each experiment and save
    
    Args:
        results: List of results, each element contains step_to_success data
        output_path: Output image path, if not specified use default path
        figsize: Figure size
    """
    if not results:
        print("No data to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color cycle
    colors = plt.cm.tab10.colors
    color_idx = 0
    
    for r in results:
        step_to_success = r.get('step_to_success', {})
        if not step_to_success:
            continue
        
        # Sort by step
        sorted_steps = sorted(step_to_success.keys())
        steps = sorted_steps
        success_values = [step_to_success[s] for s in sorted_steps]
        
        # Plot curve
        experiment_name = r['experiment_name']
        ax.plot(steps, success_values, 
                marker='o', 
                markersize=4,
                linewidth=1.5,
                color=colors[color_idx % len(colors)],
                label=experiment_name)
        color_idx += 1
    
    ax.set_xlabel('Global Step', fontsize=12)
    ax.set_ylabel('Success Once', fontsize=12)
    ax.set_title('Success Once vs Global Step', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 确定输出路径
    if output_path is None:
        output_path = 'success_once_curve.png'
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n曲线图已保存到: {output_path}")
    plt.close()


def plot_single_experiment(result: Dict, output_path: str = None, figsize: tuple = (10, 6)):
    """
    Plot success_once curve for single experiment and save
    
    Args:
        result: Single experiment result dictionary
        output_path: Output image path
        figsize: Figure size
    """
    step_to_success = result.get('step_to_success', {})
    if not step_to_success:
        print("No data to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by step
    sorted_steps = sorted(step_to_success.keys())
    steps = sorted_steps
    success_values = [step_to_success[s] for s in sorted_steps]
    
    # Plot curve
    ax.plot(steps, success_values, 
            marker='o', 
            markersize=5,
            linewidth=2,
            color='#2E86AB',
            label=result['experiment_name'])
    
    ax.set_xlabel('Global Step', fontsize=12)
    ax.set_ylabel('Success Once', fontsize=12)
    ax.set_title(f"Success Once Curve - {result['experiment_name']}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis range
    ax.set_ylim(0, max(success_values) * 1.1 if success_values else 1)
    
    # Adjust layout padding
    plt.tight_layout()
    
    # Determine output path if not specified
    if output_path is None:
        output_path = f"{result['experiment_name']}_success_once_curve.png"
    
    # Save image
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nImage saved to path: {output_path}")
    plt.close()


def save_success_once_data(results: list, output_path: str = None):
    """
    Save success_once data to CSV file
    
    Args:
        results: List of results, each element contains step_to_success data
        output_path: Output file path
    """
    if not results:
        return
    
    if output_path is None:
        output_path = 'success_once_data.csv'
    
    with open(output_path, 'w') as f:
        f.write("experiment_name,step,success_once\n")
        
        for r in results:
            step_to_success = r.get('step_to_success', {})
            experiment_name = r['experiment_name']
            
            for step in sorted(step_to_success.keys()):
                f.write(f"{experiment_name},{step},{step_to_success[step]}\n")
    
    print(f"Data saved to path: {output_path}")


def compute_curve_similarity(step_to_success_1: Dict[int, float], 
                             step_to_success_2: Dict[int, float],
                             method: str = 'pearson') -> Dict[str, float]:
    """
    Compute similarity between two curves based on success_once data
    
    Args:
        step_to_success_1: First curve step-to-success mapping
        step_to_success_2: Second curve step-to-success mapping
        method: Similarity calculation method, supported:
            - 'pearson': Pearson correlation coefficient (default)
            - 'spearman': Spearman correlation coefficient
            - 'mse': Mean squared error (MSE)
            - 'mae': Mean absolute error (MAE)
            - 'cosine': Cosine similarity
            - 'dtw': Dynamic time warping distance
            - 'all': Return all metrics
            
    Returns:
        Dictionary containing similarity metrics
    """
    # 找到共同的steps
    steps_1 = set(step_to_success_1.keys())
    steps_2 = set(step_to_success_2.keys())
    common_steps = sorted(steps_1 & steps_2)
    
    if len(common_steps) < 2:
        return {
            'method': method,
            'similarity': None,
            'common_steps': len(common_steps),
            'error': 'Not enough common steps for comparison (need at least 2)'
        }
    
    values_1 = np.array([step_to_success_1[s] for s in common_steps])
    values_2 = np.array([step_to_success_2[s] for s in common_steps])
    
    results = {'common_steps': len(common_steps)}
    
    def pearson_correlation(v1, v2):
        """Pearson correlation coefficient"""
        if np.std(v1) == 0 or np.std(v2) == 0:
            return None
        return stats.pearsonr(v1, v2)[0]
    
    def spearman_correlation(v1, v2):
        """Spearman correlation coefficient"""
        if np.std(v1) == 0 or np.std(v2) == 0:
            return None
        return stats.spearmanr(v1, v2)[0]
    
    def mse(v1, v2):
        """Mean squared error (MSE)"""
        return np.mean((v1 - v2) ** 2)
    
    def mae(v1, v2):
        """Mean absolute error (MAE)"""
        return np.mean(np.abs(v1 - v2))
    
    def cosine_similarity(v1, v2):
        """Cosine similarity"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return None
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def dtw_distance(v1, v2):
        """Dynamic time warping distance"""
        n, m = len(v1), len(v2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(v1[i - 1] - v2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # 插入
                    dtw_matrix[i, j - 1],      # 删除
                    dtw_matrix[i - 1, j - 1]   # 匹配
                )
        return dtw_matrix[n, m]
    
    # 根据method计算相似度
    if method == 'pearson':
        results['method'] = 'pearson'
        results['similarity'] = pearson_correlation(values_1, values_2)
        results['description'] = 'Pearson correlation coefficient (range [-1, 1], closer to 1 means more similar)'
    elif method == 'spearman':
        results['method'] = 'spearman'
        results['similarity'] = spearman_correlation(values_1, values_2)
        results['description'] = 'Spearman correlation coefficient (range [-1, 1], closer to 1 means more similar)'
    elif method == 'mse':
        results['method'] = 'mse'
        results['similarity'] = mse(values_1, values_2)
        results['description'] = 'Mean Squared Error (smaller means more similar)'
    elif method == 'mae':
        results['method'] = 'mae'
        results['similarity'] = mae(values_1, values_2)
        results['description'] = 'Mean Absolute Error (smaller means more similar)'
    elif method == 'cosine':
        results['method'] = 'cosine'
        results['similarity'] = cosine_similarity(values_1, values_2)
        results['description'] = 'Cosine similarity (range [-1, 1], closer to 1 means more similar)'
    elif method == 'dtw':
        results['method'] = 'dtw'
        results['similarity'] = dtw_distance(values_1, values_2)
        results['description'] = 'Dynamic Time Warping distance (smaller means more similar)'
    elif method == 'all':
        results['method'] = 'all'
        results['pearson'] = pearson_correlation(values_1, values_2)
        results['spearman'] = spearman_correlation(values_1, values_2)
        results['mse'] = mse(values_1, values_2)
        results['mae'] = mae(values_1, values_2)
        results['cosine'] = cosine_similarity(values_1, values_2)
        results['dtw'] = dtw_distance(values_1, values_2)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: pearson, spearman, mse, mae, cosine, dtw, all")
    
    return results


def compare_with_baseline(result: Dict, baseline_log_path: str, 
                          method: str = 'pearson') -> Dict:
    """   
    Args:
        result: Experiment result dictionary (contains step_to_success)
        baseline_log_path: Baseline log file path
        method: Similarity calculation method
        
    Returns:
        Dictionary containing comparison results
    """
    baseline_name, baseline_step_to_success = parse_log_file(baseline_log_path)
    
    similarity_result = compute_curve_similarity(
        result.get('step_to_success', {}),
        baseline_step_to_success,
        method=method
    )
    
    return {
        'experiment_name': result['experiment_name'],
        'baseline_name': baseline_name,
        'baseline_log_path': baseline_log_path,
        'similarity_result': similarity_result
    }


def compare_results_with_baseline(results: list, baseline_log_path: str,
                                   method: str = 'pearson') -> list:
    """
    Batch compare multiple experiment results with baseline
    
    Args:
        results: Experiment result list
        baseline_log_path: Baseline log file path
        method: Similarity calculation method
        
    Returns:
        Comparison result list
    """
    comparison_results = []
    for result in results:
        try:
            comparison = compare_with_baseline(result, baseline_log_path, method)
            comparison_results.append(comparison)
        except Exception as e:
            comparison_results.append({
                'experiment_name': result.get('experiment_name', 'unknown'),
                'baseline_log_path': baseline_log_path,
                'error': str(e)
            })
    return comparison_results


def print_comparison_results(comparison_results: list):
    """
    Print comparison results
    
    Args:
        comparison_results: Comparison result list
    """
    print("=" * 120)
    print(f"{'Experiment Name':<45} {'Baseline Name':<35} {'Similarity Method':<12} {'Similarity Value':<15} {'Common Steps':<10}")
    print("=" * 120)
    
    for c in comparison_results:
        if 'error' in c:
            print(f"{c['experiment_name']:<45} {'N/A':<35} {'N/A':<12} {'Error: ' + c['error']:<15}")
            continue
        
        sim_result = c['similarity_result']
        method = sim_result.get('method', 'N/A')
        similarity = sim_result.get('similarity')
        common_steps = sim_result.get('common_steps', 0)
        
        if similarity is not None:
            if method in ['mse', 'mae', 'dtw']:
                sim_str = f"{similarity:.6f}"
            else:
                sim_str = f"{similarity:.6f}"
        else:
            sim_str = "N/A"
        
        print(f"{c['experiment_name']:<45} {c['baseline_name']:<35} {method:<12} {sim_str:<15} {common_steps:<10}")
    
    print("=" * 120)


def plot_comparison_with_baseline(result: Dict, baseline_log_path: str,
                                   output_path: str = None, figsize: tuple = (12, 8)):
    """
    Plot comparison between experiment curve and baseline curve
    
    Args:
        result: Experiment result dictionary
        baseline_log_path: Baseline log file path
        output_path: Output image path
        figsize: Figure size
    """
    # Parse baseline
    baseline_name, baseline_step_to_success = parse_log_file(baseline_log_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot experiment curve
    step_to_success = result.get('step_to_success', {})
    if step_to_success:
        sorted_steps = sorted(step_to_success.keys())
        steps = sorted_steps
        success_values = [step_to_success[s] for s in sorted_steps]
        ax.plot(steps, success_values, 
                marker='o', markersize=4, linewidth=2,
                color='#2E86AB', label=result['experiment_name'])
    
    # Plot baseline curve
    if baseline_step_to_success:
        sorted_steps = sorted(baseline_step_to_success.keys())
        steps = sorted_steps
        success_values = [baseline_step_to_success[s] for s in sorted_steps]
        ax.plot(steps, success_values, 
                marker='s', markersize=4, linewidth=2,
                color='#E94F37', label=f'Baseline: {baseline_name}', linestyle='--')
    
    ax.set_xlabel('Global Step', fontsize=12)
    ax.set_ylabel('Success Once', fontsize=12)
    ax.set_title(f"Success Once Comparison: {result['experiment_name']} vs Baseline", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = f"{result['experiment_name']}_vs_baseline.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def save_comparison_results(comparison_results: list, output_path: str = None):
    """
    Save comparison results to CSV file
    
    Args:
        comparison_results: Comparison result list
        output_path: Output file path
    """
    if not comparison_results:
        return
    
    if output_path is None:
        output_path = 'baseline_comparison.csv'
    
    with open(output_path, 'w') as f:
        f.write("experiment_name,baseline_name,method,similarity,common_steps,baseline_log_path\n")
        
        for c in comparison_results:
            if 'error' in c:
                f.write(f"{c['experiment_name']},N/A,N/A,Error: {c['error']},0,{c['baseline_log_path']}\n")
                continue
            
            sim_result = c['similarity_result']
            similarity = sim_result.get('similarity', '')
            if similarity is not None:
                similarity = f"{similarity:.6f}"
            method = sim_result.get('method', 'N/A')
            common_steps = sim_result.get('common_steps', 0)
            f.write(f"{c['experiment_name']},{c['baseline_name']},{method},{similarity},{common_steps},{c['baseline_log_path']}\n")
    
    print(f"Comparison results saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse training logs, extract success_once metrics and check training status')
    parser.add_argument('path', type=str, help='Log file or directory path')
    parser.add_argument('--step', type=int, default=100, help='Target step (default: 100)')
    parser.add_argument('--log-filename', type=str, default='run_embodiment.log', 
                        help='Log filename (default: run_embodiment.log)')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold for check_global_step, if not specified use 10%% of total steps')
    parser.add_argument('--plot', type=str, default=None,
                        help='Plot curve and save to specified path (default: success_once_curve.png)')
    parser.add_argument('--plot-data', type=str, default=None,
                        help='Save success_once data for each step to CSV file (default: success_once_data.csv)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not plot curve')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Baseline log file path for curve similarity comparison')
    parser.add_argument('--similarity-method', type=str, default='pearson',
                        choices=['pearson', 'spearman', 'mse', 'mae', 'cosine', 'dtw', 'all'],
                        help='Similarity calculation method (default: pearson)')
    parser.add_argument('--comparison-output', type=str, default=None,
                        help='Comparison result output file path (default: baseline_comparison.csv)')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        # Process single file
        results = [process_single_log(str(path), args.step, args.threshold)]
    elif path.is_dir():
        # Process directory
        results = process_log_directory(str(path), args.step, args.log_filename, args.threshold)
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)
    
    # Print results
    print_results(results)
    
    # Output to file
    if args.output:
        with open(args.output, 'w') as f:
            f.write("experiment_name,actual_step,success_once,reached_threshold,crashed_before_threshold,found_target,total_steps,max_step,log_path\n")
            for r in results:
                f.write(f"{r['experiment_name']},{r['actual_step']},{r['success_once']},{r['reached_threshold']},{r['crashed_before_threshold']},{r['found_target']},{r['total_steps_found']},{r['max_step']},{r['log_path']}\n")
        print(f"\nResults saved to: {args.output}")
    
    # Plot curves
    if not args.no_plot:
        if len(results) == 1:
            # Single experiment, plot individual curve
            plot_single_experiment(results[0], args.plot)
        else:
            # Multiple experiments, plot comparison curves
            plot_success_once_curves(results, args.plot)
    
    # Save each step's data to CSV
    if args.plot_data:
        save_success_once_data(results, args.plot_data)
    
    # Compare with baseline
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(f"Error: baseline file {args.baseline} does not exist")
            sys.exit(1)
        
        # Calculate similarity
        comparison_results = compare_results_with_baseline(
            results, args.baseline, args.similarity_method
        )
        
        # Print comparison results
        print("\n" + "=" * 60)
        print("Baseline Comparison Results")
        print("=" * 60)
        print_comparison_results(comparison_results)
        
        # Save comparison results
        if args.comparison_output:
            save_comparison_results(comparison_results, args.comparison_output)
        else:
            save_comparison_results(comparison_results)
        
        # Plot comparison charts
        if not args.no_plot:
            for i, result in enumerate(results):
                output_path = f"{result['experiment_name']}_vs_baseline.png" if args.plot is None else f"{args.plot.rsplit('.', 1)[0]}_{i}.png" if len(results) > 1 else args.plot
                try:
                    plot_comparison_with_baseline(result, args.baseline, output_path)
                except Exception as e:
                    print(f"Failed to plot comparison chart: {e}")


if __name__ == '__main__':
    main()