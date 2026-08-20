"""Ring attractor network model from Noorman et al., 2024, Nat. Neurosci.

Implements the single-ring network of N orientation-tuned neurons described
in the "Model overview" section of the Methods (Eq. 1):

    tau * dh_j/dt = -h_j + (1/N) * sum_k (W_jk^sym + v_in * W_jk^asym) * phi(h_k) + c_ff

with symmetric (local excitation / broad inhibition) connectivity
    W_jk^sym = J_I + J_E * cos(theta_j - theta_k)
and antisymmetric, velocity-modulated connectivity
    W_jk^asym = sin(theta_j - theta_k)
"""

import warnings

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display


def preferred_headings(N):
    """Evenly spaced preferred headings theta_j for a ring of N neurons.

    Args:
        N: number of neurons (computational units) in the ring.

    Returns:
        Array of shape (N,) of preferred headings in [0, 2*pi).
    """
    return np.arange(N) * 2 * np.pi / N


def symmetric_weights(N, JE, JI, JE_noise=None, JI_noise=None):
    """Symmetric connectivity W_jk^sym = JI + JE * cos(theta_j - theta_k).

    JE and JI parametrize the strength of local excitation and broad
    (uniform) inhibition, respectively.

    Args:
        N: number of neurons.
        JE: local excitation strength.
        JI: broad inhibition strength.
        JE_noise: optional (N, N) array added elementwise to JE before it
            multiplies cos(theta_j - theta_k), representing per-synapse
            heterogeneity in local excitation.
        JI_noise: optional (N, N) array added elementwise to JI, representing
            per-synapse heterogeneity in broad inhibition.

    Returns:
        Array of shape (N, N).
    """
    theta = preferred_headings(N)
    dtheta = theta[:, None] - theta[None, :]
    JE_mat = JE if JE_noise is None else JE + JE_noise
    JI_mat = JI if JI_noise is None else JI + JI_noise
    return JI_mat + JE_mat * np.cos(dtheta)


def sample_weight_noise(N, rho, rng=None):
    """Sample an (N, N) matrix of i.i.d. noise, uniform in [-rho, rho].

    Args:
        N: number of neurons.
        rho: noise magnitude; entries are drawn uniformly from [-rho, rho].
        rng: optional numpy.random.Generator for reproducibility; a fresh
            default generator is used if not given.

    Returns:
        Array of shape (N, N).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-rho, rho, size=(N, N))


def asymmetric_weights(N):
    """Antisymmetric, velocity-modulated connectivity W_jk^asym = sin(theta_j - theta_k).

    Args:
        N: number of neurons.

    Returns:
        Array of shape (N, N).
    """
    theta = preferred_headings(N)
    dtheta = theta[:, None] - theta[None, :]
    return np.sin(dtheta)


def plot_weight_matrix(N, JE=None, JI=None, symmetric=True, ax=None,
                        JE_noise=None, JI_noise=None):
    """Plot a recurrent connectivity matrix as a diverging heatmap.

    Args:
        N: number of neurons.
        JE: local excitation strength; required if symmetric=True.
        JI: broad inhibition strength; required if symmetric=True.
        symmetric: if True, plot the symmetric connectivity W^sym (local
            excitation + broad inhibition, requires JE and JI); if False,
            plot the antisymmetric, velocity-modulated connectivity W^asym
            (independent of JE and JI).
        ax: optional matplotlib Axes to plot into; a new figure/axes is
            created if not given.
        JE_noise: optional (N, N) array of per-synapse noise added to JE
            (only used when symmetric=True; see symmetric_weights).
        JI_noise: optional (N, N) array of per-synapse noise added to JI
            (only used when symmetric=True; see symmetric_weights).

    Returns:
        The matplotlib Axes containing the plot.
    """
    if symmetric:
        if JE is None or JI is None:
            raise ValueError("JE and JI must be given when symmetric=True")
        W = symmetric_weights(N, JE, JI, JE_noise=JE_noise, JI_noise=JI_noise)
        title = f"Symmetric connectivity $W^{{sym}}$ (JE={JE:.02f}, JI={JI:.02f})"
    else:
        W = asymmetric_weights(N)
        title = "Antisymmetric connectivity $W^{asym}$"

    if ax is None:
        _, ax = plt.subplots()

    vmax = max(np.max(np.abs(W)), 1e-12)
    im = ax.imshow(W, cmap="bwr", vmin=-vmax, vmax=vmax)

    theta = preferred_headings(N)
    tick_labels = [f"{th:.2f}" for th in theta]
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel(r"Preferred heading $\theta_k$ (rad)")
    ax.set_ylabel(r"Preferred heading $\theta_j$ (rad)")
    ax.set_title(title)

    ax.figure.colorbar(im, ax=ax, label="Weight")

    return ax


def threshold_linear(h):
    """Threshold-linear (rectified linear) transfer function phi(h) = [h]_+."""
    return np.maximum(h, 0.0)


def optimal_JE(N, Nact, warn_threshold=20.0):
    """Optimal local excitation J_E* that flattens the energy landscape (Eq. 11).

    For a network of size N, there are N - 3 such optimal values, one for
    each choice of the number of active neurons Nact in [2, N - 2]. J_E*
    is smallest (and best-behaved numerically) when Nact is near N/2, and
    grows rapidly as Nact approaches either edge of its range (2 or N - 2):
    e.g. for N=24, J_E* is ~4 at Nact=12 but ~179 at Nact=3. A very large
    J_E* makes the resulting dynamics numerically stiff, causing simulations
    to run far more slowly, and requires correspondingly strong inhibition
    (J_I) to remain stable at all -- an Nact carried over from a smaller
    network (e.g. Nact=3, appropriate for N=6 where it equals N/2) is a
    common way to hit this by accident when scaling N up.

    Args:
        N: number of neurons.
        Nact: number of active neurons maintaining the bump (2 <= Nact <= N - 2).
        warn_threshold: if the resulting J_E* exceeds this value, a warning
            is issued suggesting an Nact closer to N/2. Set to None to
            disable this check.

    Returns:
        Optimal value of local excitation J_E*.
    """
    n_tilde = Nact - N / 2
    inv_JE = 0.25 + (1.0 / (2 * N)) * (n_tilde + np.sin(2 * np.pi * n_tilde / N) / np.sin(2 * np.pi / N))
    JE = 1.0 / inv_JE
    if warn_threshold is not None and JE > warn_threshold:
        warnings.warn(
            f"optimal_JE(N={N}, Nact={Nact}) = {JE:.1f} is far from the "
            f"well-behaved regime (smallest, ~4, at Nact=N/2={N / 2:.0f}); "
            "such large JE* values require correspondingly strong JI to "
            "remain stable and make simulations numerically stiff/slow. "
            "Consider choosing Nact closer to N/2.",
            stacklevel=2,
        )
    return JE


def find_JI_for_amplitude(N, JE, cff=1.0, target_amplitude=0.2, n_samples=2000,
                           max_JI_over_JE=50.0, divergence_threshold=50.0,
                           tol=1e-2):
    """Select JI to produce a bump of a desired full amplitude, given JE.

    Follows the "Stable parameter regime" procedure in the Methods: samples
    bump widths/orientations along the contour JE * f_even(w, psi) = 1, then
    picks JI so that the resulting bump amplitude A = H0 + a matches
    target_amplitude, subject to JI < JI_bound.

    This is a light-weight numerical approximation to the procedure
    described in the paper (which relies on the closed-form Fourier-mode
    equations derived in the Supplementary Note); here we instead directly
    search over JI by simulating the network to steady state, which is more
    robust for classroom use.

    Bump amplitude decreases monotonically as |JI| grows (stronger broad
    inhibition suppresses activity), so JI is found by bisection on |JI|
    rather than by scanning a fixed or pre-sampled grid: a scan either
    wastes most of its budget on uninformative magnitudes, or (if scaled up
    to cover the very large |JI| that a poorly chosen, large JE can require
    -- see optimal_JE) becomes slow, since large weight magnitudes make the
    network dynamics numerically stiff regardless of whether they are
    excitatory or inhibitory. Bisection instead needs only a couple dozen
    simulations to converge, however wide the range turns out to be.
    Candidates whose simulated activity diverges are treated as if their
    amplitude were infinite (i.e. still above the target), which is
    consistent with -- and does not break -- the monotonic bisection.

    Args:
        N: number of neurons.
        JE: local excitation strength (fixed).
        cff: constant feedforward input.
        target_amplitude: desired peak activity (approx. full bump amplitude).
        n_samples: maximum number of bisection iterations (kept under its
            original name for backwards compatibility; bisection converges
            in a few dozen iterations at most, so this is effectively just
            a generous upper bound).
        max_JI_over_JE: the initial JI search range extends from 0 down to
            -max_JI_over_JE * JE (with a floor of -30, matching the original
            fixed range for small JE); doubled as needed if the target
            amplitude isn't reached within it.
        divergence_threshold: candidates whose simulated bump amplitude
            exceeds this value (or is non-finite) are treated as unstable
            and discarded (equivalent to an infinite amplitude).
        tol: bisection stops once the bracket on |JI| is narrower than this.

    Returns:
        JI value (float) whose steady-state bump amplitude is closest to
        target_amplitude.

    Raises:
        RuntimeError: if the search range cannot be expanded to find any
            magnitude of JI that is both stable and at/below the target
            amplitude.
    """
    def amplitude_at(JI_mag):
        try:
            r = simulate_bump(N, JE, -JI_mag, cff=cff, vin=0.0, t_max=20.0,
                               psi0=0.0, return_trajectory=False)
        except Exception:
            return np.inf  # solver failed outright (e.g. numerical overflow)
        amp = r.max() - r.min()
        if not np.isfinite(amp) or amp > divergence_threshold:
            return np.inf  # unstable/diverging; treat as "infinite" amplitude
        return amp

    lo, hi = 0.01, max(30.0, max_JI_over_JE * JE)
    amp_lo = amplitude_at(lo)
    if amp_lo <= target_amplitude:
        return -lo  # even minimal inhibition already meets the target

    amp_hi = amplitude_at(hi)
    while amp_hi > target_amplitude:
        hi *= 2
        if hi > 1e8:
            raise RuntimeError(
                f"Could not find a JI that stabilizes JE={JE} at amplitude "
                f"{target_amplitude}; try increasing max_JI_over_JE."
            )
        amp_hi = amplitude_at(hi)

    for _ in range(n_samples):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if amplitude_at(mid) > target_amplitude:
            lo = mid
        else:
            hi = mid
    return -hi


def init_bump(N, psi0, width=None, amplitude=1.0):
    """Initialize a cosine-shaped bump of activity centered at psi0.

    Args:
        N: number of neurons.
        psi0: initial bump orientation (rad).
        width: unused placeholder kept for interface symmetry with the
            paper's (a, w, psi) bump parametrization; the single-ring model
            here uses a raw cosine bump instead.
        amplitude: peak activity of the initial bump.

    Returns:
        Array of shape (N,) giving initial input activity h_0.
    """
    theta = preferred_headings(N)
    return amplitude * np.cos(theta - psi0)


def network_rhs(t, h, N, JE, JI, cff, vin, tau, phi=threshold_linear,
                 JE_noise=None, JI_noise=None):
    """Right-hand side of the network dynamics (Eq. 1).

    tau * dh_j/dt = -h_j + (1/N) sum_k (W_sym_jk + vin * W_asym_jk) * phi(h_k) + cff

    Args:
        t: time (unused; dynamics are autonomous given constant vin).
        h: current input activity, shape (N,).
        N: number of neurons.
        JE: local excitation strength.
        JI: broad inhibition strength.
        cff: constant feedforward input.
        vin: angular velocity input (constant).
        tau: single-neuron time constant.
        phi: nonlinear transfer function (default: threshold-linear).
        JE_noise: optional (N, N) array of per-synapse noise added to JE.
        JI_noise: optional (N, N) array of per-synapse noise added to JI.

    Returns:
        dh/dt, shape (N,).
    """
    Wsym = symmetric_weights(N, JE, JI, JE_noise=JE_noise, JI_noise=JI_noise)
    Wasym = asymmetric_weights(N)
    r = phi(h)
    total_input = (Wsym + vin * Wasym) @ r / N
    return (-h + total_input + cff) / tau


def animate_network_activity(t, h, vin, phi=threshold_linear, interval=50,
                              ylabel="Firing rate"):
    """Animate each neuron's activity over the course of a simulation.

    Draws a bar plot of the firing rate phi(h_j) for each neuron j, updated
    frame-by-frame over the simulated trajectory, with the current time and
    v_in shown in the title.

    Args:
        t: time points, shape (n_steps,), as returned by simulate_bump.
        h: input activity trajectory, shape (n_steps, N), as returned by
            simulate_bump.
        vin: angular velocity input used for the simulation. Either a scalar
            (constant velocity, shown unchanged in every frame) or an array
            of shape (n_steps,) giving the velocity at each timestep.
        phi: nonlinear transfer function used to convert h into firing rate
            (default: threshold-linear).
        interval: delay between animation frames, in ms.
        ylabel: y-axis label.

    Returns:
        None. Displays the animation inline (for use in a Jupyter notebook).
    """
    t = np.asarray(t)
    h = np.asarray(h)
    n_steps, N = h.shape
    r = phi(h)
    vin_per_frame = np.broadcast_to(vin, (n_steps,))

    neuron_idx = np.arange(N)
    fig, ax = plt.subplots()
    ax.set_ylim(0, max(r.max(), 1e-6) * 1.1)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel(ylabel)
    bars = ax.bar(neuron_idx, r[0])

    def _updatefig(frame):
        for bar, height in zip(bars, r[frame]):
            bar.set_height(height)
        ax.set_title(f"t = {t[frame]:.2f} s, v_in = {vin_per_frame[frame]:.3f} rad/s")
        return bars

    anim = animation.FuncAnimation(
        fig, _updatefig, interval=interval, frames=n_steps, blit=False)
    html = HTML(anim.to_jshtml())
    display(html)
    plt.close()


def simulate_bump(N, JE, JI, cff=1.0, vin=0.0, tau=0.1, t_max=3.0, dt=0.01,
                   psi0=0.0, h0=None, phi=threshold_linear,
                   return_trajectory=True, animate=False,
                   JE_noise=None, JI_noise=None, method="RK45"):
    """Simulate the ring attractor network's dynamics (Eq. 1).

    Args:
        N: number of neurons.
        JE: local excitation strength.
        JI: broad inhibition strength.
        cff: constant feedforward input.
        vin: (constant) angular velocity input.
        tau: single-neuron time constant.
        t_max: total simulation time (s).
        dt: sampling timestep for the returned trajectory (s).
        psi0: initial bump orientation, used if h0 is not given.
        h0: optional explicit initial condition, shape (N,). Overrides psi0.
        phi: nonlinear transfer function (default: threshold-linear).
        return_trajectory: if True, return the full (t, h) trajectory;
            if False, return only the final activity vector h(t_max).
        animate: if True, display an animation of each neuron's activity
            over the course of the simulation (see animate_network_activity).
            Requires return_trajectory=True.
        JE_noise: optional (N, N) array of per-synapse noise added to JE for
            the duration of this simulation (see symmetric_weights); use
            sample_weight_noise to generate one.
        JI_noise: optional (N, N) array of per-synapse noise added to JI for
            the duration of this simulation (see symmetric_weights); use
            sample_weight_noise to generate one.
        method: scipy.integrate.solve_ivp integration method. The default
            explicit "RK45" is fast for the well-behaved parameter regime
            (JE near its optimal value for Nact close to N/2). Very large
            JE/JI (e.g. from an Nact far from N/2 in optimal_JE) make the
            dynamics numerically stiff, so RK45 needs very many tiny steps
            and can appear to hang; in that regime an implicit method such
            as "BDF" is far faster, though it is more prone to raising a
            solver error outright if the activity nonetheless overflows.

    Returns:
        If return_trajectory: (t, h) where t has shape (n_steps,) and h has
            shape (n_steps, N).
        Else: h(t_max), shape (N,).
    """
    if h0 is None:
        h0 = init_bump(N, psi0)

    n_steps = int(round(t_max / dt)) + 1
    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(network_rhs, (0.0, t_max), h0, t_eval=t_eval,
                     args=(N, JE, JI, cff, vin, tau, phi, JE_noise, JI_noise),
                     method=method)

    if return_trajectory:
        if animate:
            animate_network_activity(sol.t, sol.y.T, vin, phi=phi)
        return sol.t, sol.y.T
    return sol.y[:, -1]


def bump_orientation(h, N=None):
    """Estimate the bump orientation via the population vector average (PVA).

    Args:
        h: activity, shape (N,) or (n_steps, N).
        N: number of neurons; inferred from h if not given.

    Returns:
        Bump orientation(s) in (-pi, pi]: scalar if h is 1D, else shape
        (n_steps,).
    """
    h = np.asarray(h)
    if N is None:
        N = h.shape[-1]
    theta = preferred_headings(N)
    r = threshold_linear(h)
    z = r @ np.exp(1j * theta)
    return np.angle(z)


def bump_amplitude(h, N=None):
    """Estimate the (relative) bump amplitude via the PVA vector length.

    Args:
        h: activity, shape (N,) or (n_steps, N).
        N: number of neurons; inferred from h if not given.

    Returns:
        Bump amplitude(s), normalized so a fully localized single-neuron
        bump has amplitude 1.
    """
    h = np.asarray(h)
    if N is None:
        N = h.shape[-1]
    theta = preferred_headings(N)
    r = threshold_linear(h)
    z = r @ np.exp(1j * theta)
    denom = r.sum(axis=-1)
    denom = np.where(denom == 0, 1.0, denom)
    return np.abs(z) / denom
