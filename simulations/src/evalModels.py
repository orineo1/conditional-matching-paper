import dist_utils
import torch
from functools import partial
from tqdm import tqdm
import LossFunctions


def evaluate_conditional_model(
        model,
        mu_list,
        Sigma_list,
        alpha,
        condition_on,
        device,
        num_evaluations=500,
        nsamples=5000,
        num_projections=1000,
        extract_generated_only=True,  # Set False for CM models
        model_type='diffusion',  # 'diffusion', 'cm', 'fm', or 'super_diffusion'
        reverse_distribution=False  # If True, flip mu and reverse Sigma indices
):
    """
    Evaluate conditional model using SWD metric.

    Args:
        model: Trained conditional model (Diffusion, ConsistencyModel, or FlowMatching)
        mu_list: List of means for MoG
        Sigma_list: List of covariances for MoG
        alpha: Mixture weights
        condition_on: Number of features to condition on
        device: torch device
        num_evaluations: Number of evaluation iterations
        nsamples: Number of samples per evaluation
        num_projections: Number of projections for SWD
        extract_generated_only: Whether to extract only generated features (not conditioning)
        model_type: Type of model ('diffusion', 'cm', 'fm', 'super_diffusion')
        reverse_distribution: If True, reverse the order of features in mu and Sigma

    Returns:
        avg_swd: Average SWD
        std_swd: Standard deviation of SWD
        swd_list: List of all SWD values
    """
    swd_list = []

    # Apply reversal if needed
    if reverse_distribution:
        mu_list_eval = [p.flip(0) for p in mu_list]
        idx = torch.arange(Sigma_list[0].size(0) - 1, -1, -1)
        Sigma_list_eval = [s[idx][:, idx] for s in Sigma_list]
        mu_list_eval = [mu.float() for mu in mu_list_eval]
        Sigma_list_eval = [cov.float() for cov in Sigma_list_eval]
        alpha_eval = alpha.float()
    else:
        mu_list_eval = mu_list
        Sigma_list_eval = Sigma_list
        alpha_eval = alpha

    # Create data generator
    data_generator = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list_eval,
        variances=Sigma_list_eval,
        weights=alpha_eval,
        kernel_func=None
    )

    with torch.no_grad():
        for i in tqdm(range(num_evaluations)):
            # Sample from joint distribution
            joint_sample = data_generator(1, device=device)

            # Extract conditioning vector
            condition_vector = joint_sample[:, :condition_on]

            # Sample from model based on type
            if model_type == 'diffusion':
                model_samples, _, _ = model.sample(
                    nsamples=nsamples,
                    condition_x=condition_vector,
                    device=device
                )
            elif model_type == 'cm':
                model_samples, _, _ = model.sample(
                    nsamples=nsamples,
                    condition_x=condition_vector.float(),
                    device=device
                )
            elif model_type == 'fm':
                model_samples = model.sample(
                    nsamples,
                    condition_vector.float()
                )
            elif model_type == 'super_diffusion':
                condition_vector_rep = condition_vector.repeat(nsamples, 1)
                # NEW: Returns only 2 values, then extract last timestep
                model_samples, _ = model.sample(
                    nsamples=nsamples,
                    cond_val=condition_vector_rep,
                    track_gradients=False,
                    diffusion_steps=1e-2,
                    compute_log_likelihood=True
                )
                model_samples = model_samples[:, -1, :]
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Extract only generated part if needed
            if extract_generated_only:
                model_samples = model_samples[:, model.condition_on:]

            # Compute true conditional distribution
            mu_cond, Sigma_cond = dist_utils.compute_conditionals(
                mu_list_eval, Sigma_list_eval, condition_vector.reshape(-1, 1)
            )

            alpha_cond = dist_utils.compute_alpha(
                mu_list_eval, Sigma_list_eval, alpha_eval, condition_vector.flatten()
            )

            # Sample from true conditional
            true_samples = dist_utils.generate_mog_samples_not_differentiable(
                nsamples, mu_cond, Sigma_cond, alpha_cond
            ).float().to(device)
            # print(f" Sigma_cond.shape:{ Sigma_cond.shape}")
            # Compute SWD
            swd_raw, swd_norm = LossFunctions.sliced_wasserstein_distance(
                model_samples,
                true_samples,
                num_projections=num_projections,
                p=2,
                mu_list_sliced=mu_cond,
                Sigma_list_sliced=Sigma_cond,
                alpha=alpha_cond,
                Flag=False
            )

            swd_list.append(swd_norm)

    avg_swd = np.mean(swd_list)
    std_swd = np.std(swd_list)

    print(f"Average Conditional SWD: {avg_swd:.6f} ± {std_swd:.6f}")

    return avg_swd, std_swd, swd_list


def evaluate_unconditional_model(
        model,
        mu_list,
        Sigma_list,
        alpha,
        condition_on,
        device,
        nsamples=10000,
        num_projections=1000,
        model_type='diffusion',  # 'diffusion' or 'fm'
        num_evaluations=1  # Number of evaluation iterations
):
    """
    Evaluate unconditional model using SWD metric.

    Args:
        model: Trained unconditional model
        mu_list: List of means for MoG
        Sigma_list: List of covariances for MoG
        alpha: Mixture weights
        condition_on: Number of features in conditioning dimension
        device: torch device
        nsamples: Number of samples to generate
        num_projections: Number of projections for SWD
        model_type: Type of model ('diffusion' or 'fm')
        num_evaluations: Number of evaluation iterations

    Returns:
        avg_swd_raw: Average raw SWD value
        avg_swd_norm: Average normalized SWD value
        swd_raw_list: List of all raw SWD values
        swd_norm_list: List of all normalized SWD values
    """
    # OLD: Single evaluation
    # NEW: Multiple evaluations with lists
    swd_raw_list = []
    swd_norm_list = []

    # OLD: No loop
    # NEW: Loop with tqdm
    with torch.no_grad():
        for i in tqdm(range(num_evaluations)):
            # Sample from model
            if model_type == 'diffusion':
                samples_uncond, _, _ = model.sample(
                    nsamples=nsamples,
                    condition_x=None,
                    device=device
                )
            elif model_type == 'fm':
                samples_uncond = model.sample(
                    num_sample=nsamples,
                    y=None,
                )
                samples_uncond = samples_uncond.cpu().detach()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Generate ground truth samples
            data_generator = partial(
                dist_utils.generate_mog_samples_not_differentiable,
                means=mu_list,
                variances=Sigma_list,
                weights=alpha,
                kernel_func=lambda X: X[:, :condition_on]
            )
            Y_true = data_generator(nsamples, device=device)

            if model_type == 'fm':
                Y_true = Y_true.cpu().detach()

            # Slice means and covariances to match sample dimension
            sample_dim = samples_uncond.shape[1]
            mu_list_sliced = [mu[:sample_dim] for mu in mu_list]
            Sigma_list_sliced = [Sigma[:sample_dim, :sample_dim] for Sigma in Sigma_list]

            # Compute SWD
            swd_raw, swd_norm = LossFunctions.sliced_wasserstein_distance(
                samples_uncond,
                Y_true,
                num_projections=num_projections,
                p=2,
                mu_list_sliced=mu_list_sliced,
                Sigma_list_sliced=Sigma_list_sliced,
                alpha=alpha,
                Flag=False
            )

            swd_raw_list.append(swd_raw)
            swd_norm_list.append(swd_norm)

    avg_swd_raw = np.mean(swd_raw_list)
    avg_swd_norm = np.mean(swd_norm_list)
    std_swd_norm = np.std(swd_norm_list)

    print(f"\nAverage Unconditional SWD: {avg_swd_norm:.6f} ± {std_swd_norm:.6f}")

    return avg_swd_raw, avg_swd_norm, swd_raw_list, swd_norm_list


def create_model_evaluation_summary(
    models_dict,
    mu_list,
    Sigma_list,
    alpha,
    condition_on,
    device,
    num_evaluations=500,
    nsamples=5000,
    num_projections=1000
):
    results = {
        'Model': [],
        'Task': [],
        'SWD_normalized': [],
        'SWD_raw': [],
    }

    total_dim = mu_list[0].shape[0]

    for model_name, config in models_dict.items():
        print(f"\nEvaluating {model_name}...")

        if config['task'] == 'P(X)':
            # OLD: swd_raw, swd_norm = evaluate_unconditional_model(...)
            # NEW: Returns lists with multiple evaluations
            avg_swd_raw, avg_swd_norm, swd_raw_list, swd_norm_list = evaluate_unconditional_model(
                model=config['model'],
                mu_list=mu_list,
                Sigma_list=Sigma_list,
                alpha=alpha,
                condition_on=condition_on,
                device=device,
                nsamples=nsamples,
                num_projections=num_projections,
                model_type=config['type'],
                num_evaluations=num_evaluations
            )

            for swd_raw, swd_norm in zip(swd_raw_list, swd_norm_list):
                results['Model'].append(model_name)
                results['Task'].append(config['task'])
                results['SWD_normalized'].append(swd_norm)
                results['SWD_raw'].append(swd_raw)
        else:
            # Conditional models
            if config.get('reverse_distribution', False):
                model_condition_on = total_dim - condition_on
            else:
                model_condition_on = condition_on

            avg_swd, std_swd, swd_list = evaluate_conditional_model(
                model=config['model'],
                mu_list=mu_list,
                Sigma_list=Sigma_list,
                alpha=alpha,
                condition_on=model_condition_on,
                device=device,
                num_evaluations=num_evaluations,
                nsamples=nsamples,
                num_projections=num_projections,
                extract_generated_only=config.get('extract_generated_only', True),
                model_type=config['type'],
                reverse_distribution=config.get('reverse_distribution', False)
            )

            for swd_val in swd_list:
                results['Model'].append(model_name)
                results['Task'].append(config['task'])
                results['SWD_normalized'].append(swd_val)
                results['SWD_raw'].append(swd_val)

    df = pd.DataFrame(results)
    summary = df.groupby(['Model', 'Task']).agg({
        'SWD_normalized': ['mean', 'std', 'min', lambda x: x.quantile(0.25), lambda x: x.quantile(0.5),
                           lambda x: x.quantile(0.75), 'max'],
        'SWD_raw': ['mean', 'std', 'min', lambda x: x.quantile(0.25), lambda x: x.quantile(0.5),
                    lambda x: x.quantile(0.75), 'max']
    }).round(6)

    # CHANGED: rename percentile columns for clarity
    summary.columns = [
        'SWD_norm_mean', 'SWD_norm_std', 'SWD_norm_min', 'SWD_norm_25%', 'SWD_norm_50%', 'SWD_norm_75%', 'SWD_norm_max',
        'SWD_raw_mean', 'SWD_raw_std', 'SWD_raw_min', 'SWD_raw_25%', 'SWD_raw_50%', 'SWD_raw_75%', 'SWD_raw_max'
    ]

    # summary = df.groupby(['Model', 'Task']).agg({
    #     'SWD_normalized': ['mean', 'std', 'min', 'max'],
    #     'SWD_raw': ['mean', 'std']
    # }).round(6)

    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)
    print(summary)
    print("\n")

    return df, summary


#### Summary Tables for optimization
import pandas as pd

# Convert tensors to numpy (detach gradients first)
def to_numpy(loss_list):
    return np.array([l.detach().cpu().item() if torch.is_tensor(l) else l for l in loss_list])


import numpy as np

def create_summary_table(indices_dict,
                         norm_swd_dict,
                         swd_dict,
                         l1_dict,
                         times_dict,
                         title="Metrics Summary"):
    """
    Create summary statistics table for given indices.

    Parameters:
    -----------
    indices_dict : dict
        {'method_name': [list of indices]}
    norm_swd_dict : dict
        {'method_name': [normalized SWD values]}
    swd_dict : dict
        {'method_name': [regular SWD values]}
    l1_dict : dict
        {'method_name': [L1 distance values]}
    times_dict : dict
        {'method_name': [time values]}
    title : str
        Title for the table
    """
    methods = []
    swd_norm_vals = []
    swd_reg_vals = []
    l1_vals = []
    time_vals = []

    for method, indices in indices_dict.items():
        n = len(indices)
        methods.extend([method] * n)
        swd_norm_vals.extend([norm_swd_dict[method][i] for i in indices])
        swd_reg_vals.extend([swd_dict[method][i] for i in indices])
        l1_vals.extend([l1_dict[method][i] for i in indices])
        time_vals.extend([times_dict[method][i] for i in indices])

    data = {
        'Method': methods,
        'SWD': swd_norm_vals,
        'SWD_regular': swd_reg_vals,
        'L1': l1_vals,
        'Time': time_vals
    }

    df = pd.DataFrame(data)

    # OLD: Using custom functions that don't work properly
    # CHANGED: Use numpy percentile function directly in aggregation
    summary = df.groupby('Method').agg({
        'SWD': ['mean', 'std', ('25%', lambda x: np.percentile(x, 25)),
                ('50%', lambda x: np.percentile(x, 50)),
                ('75%', lambda x: np.percentile(x, 75))],
        'SWD_regular': ['mean', 'std', ('25%', lambda x: np.percentile(x, 25)),
                        ('50%', lambda x: np.percentile(x, 50)),
                        ('75%', lambda x: np.percentile(x, 75))],
        'L1': ['mean', 'std', ('25%', lambda x: np.percentile(x, 25)),
               ('50%', lambda x: np.percentile(x, 50)),
               ('75%', lambda x: np.percentile(x, 75))],
        'Time': ['mean', 'std', ('25%', lambda x: np.percentile(x, 25)),
                 ('50%', lambda x: np.percentile(x, 50)),
                 ('75%', lambda x: np.percentile(x, 75))]
    }).round(4)

    # Flatten multi-level column names
    summary.columns = ['_'.join(col).strip() if col[1] else col[0]
                       for col in summary.columns.values]

    print("=" * 70)
    print(title)
    print("=" * 70)
    print(summary)
    print()

    # Time statistics with percentiles
    print("=" * 70)
    print(f"{title} - Time Statistics (seconds)")
    print("=" * 70)
    time_stats = df.groupby('Method')['Time'].agg([
        'mean', 'std',
        ('25%', lambda x: np.percentile(x, 25)),
        ('50%', lambda x: np.percentile(x, 50)),
        ('75%', lambda x: np.percentile(x, 75))
    ]).round(2)
    print(time_stats)
    print("\n")

    return df, summary
