
from pyomo.environ import *

# We import the two concrete battery models as-is.
from block_battery import battery_v2, battery_v3

def _ensure_tmp_param(bat, time):
    """Ensure a mutable temperature Param 'tmp' exists on the battery block."""
    if not hasattr(bat, 'tmp'):
        # Default 25°C if not provided, same indexing as 'time'
        try:
            bat.tmp = Param(time, mutable=True, default=25)
        except Exception:
            # fallback to scalar Param if time is not a valid set here
            bat.tmp = Param(mutable=True, default=25)

def _normalize_interfaces_v2(bat, time):
    """Add a few convenience components so 'main' can use a unified API.

    Ensures the following exist with the same semantics as V3:
    - emax[t] style (via an Expression over time)
    - emax0: Expression returning the initial (scalar) capacity
    - e_loss[t]: Var-like (Expression) equal to 0 for all t (no aging modeled in V2)
    - emax_series[t]: Expression equal to the (constant) capacity for all t
    - tmp: Param(time) (created if missing)
    """
    _ensure_tmp_param(bat, time)

    # emax0: capacity scalar for V2 (bat.emax is a scalar Var in V2)
    if not hasattr(bat, 'emax0'):
        bat.emax0 = Expression(expr=bat.emax)

    # e_loss: no aging in V2 -> zero expression (indexed by time)
    if not hasattr(bat, 'e_loss'):
        @bat.Expression(time)
        def e_loss(m, t):
            return 0.0

    # emax_series[t]: expose a time-indexed capacity for plotting & constraints that expect indexing
    if not hasattr(bat, 'emax_series'):
        @bat.Expression(time)
        def emax_series(m, t):
            return bat.emax

    # Provide a 'soc' if not present (V2 already has it)
    if not hasattr(bat, 'soc'):
        @bat.Expression(time)
        def soc(m, t):
            return 100 * m.e[t] / bat.emax

def _normalize_interfaces_v3(bat, time):
    """For V3 we already have time-indexed emax and e_loss; we just create helpers."""
    _ensure_tmp_param(bat, time)

    # emax0 : take capacity at first time index (often t=0)
    if not hasattr(bat, 'emax0'):
        first_t = list(time)[0] if hasattr(time, '__iter__') else 0
        @bat.Expression()
        def emax0(m):
            return m.emax[first_t]

    # emax_series : alias to emax[t] for clarity/consistency
    if not hasattr(bat, 'emax_series'):
        @bat.Expression(time)
        def emax_series(m, t):
            return m.emax[t]

    # e_loss already defined in V3

def make_battery(bat, model=3, **options):
    """Factory to build a battery block with a stable interface.

    Args:
        bat: a Pyomo Block provided by caller.
        model: 2 -> battery_v2 ; 3 -> battery_v3
        **options: forwarded to the underlying battery function.

    Exposed unified attributes after construction:
        - tmp (Param(time)): temperature
        - emax0 (Expression): capacity at t=0 (or scalar for V2)
        - emax_series[t] (Expression): capacity trajectory vs time
        - e_loss[t] (Expression/Var): aging loss (0 for V2)
        - soc[t] (Expression): state of charge (%)  [provided by underlying model]
        - p[t], pc[t], pd[t] : standard power vars
        - cost_inv, cost_opex : Params kept as-is
    """
    time = options.get('time', RangeSet(0,1))

    if model == 2:
        bat = battery_v2(bat, **options)
        _normalize_interfaces_v2(bat, time)
    elif model == 3:
        bat = battery_v3(bat, **options)
        _normalize_interfaces_v3(bat, time)
    else:
        raise ValueError(f"Unsupported battery model id: {model}. Use 2 or 3.")

    return bat
