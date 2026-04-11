from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchdiffeq import odeint
from flow_matching.utils import gradient
from types import MethodType


def add_compute_likelihood_allow_grad(solver):
    """
    Add the compute_likelihood_allow_grad method to an ODESolver instance.

    Args:
        solver: An instance of ODESolver from flow_matching.
    """

    def compute_likelihood_allow_grad(
            solver,
            x_1: Tensor,
            log_p0: Callable[[Tensor], Tensor],
            step_size: Optional[float],
            method: str = "euler",
            atol: float = 1e-5,
            rtol: float = 1e-5,
            time_grid: Tensor = torch.tensor([1.0, 0.0]),  # Default is just start and end
            return_intermediates: bool = False,
            exact_divergence: bool = False,
            enable_grad: bool = False,
            Train_consistency=False,  # New parameter to return all log dets
            **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        assert time_grid[0] == 1.0 and time_grid[-1] == 0.0, \
            f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return solver.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):  # Ensure gradients are computed
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    # Compute exact divergence without detaching
                    div = 0
                    for i in range(ut.flatten(1).shape[1]):
                        div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                else:
                    # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    ut_dot_z = torch.einsum("ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1))
                    grad_ut_dot_z = gradient(ut_dot_z, xt, create_graph=True)
                    div = torch.einsum("ij,ij->i", grad_ut_dot_z.flatten(start_dim=1), z.flatten(start_dim=1))

            return ut, div

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):  # Enable backpropagation
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if Train_consistency:
            # Return solution and all log determinants (shape [time_steps, batch_size])
            return sol, log_det
        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]

    solver.compute_likelihood_allow_grad = MethodType(compute_likelihood_allow_grad, solver)
    return solver