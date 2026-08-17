"""Validate a LISFLOOD-FP .par before spending a job on it.

Written after a nested 4 m run burned eighteen diagnostics on a silent misconfiguration.
The par had `latlong` but not `sgc_enable`. LISFLOOD refuses that combination in pars.cpp:

    if (Statesptr->latlong == ON && Statesptr->SGC == OFF && verbose == ON)
    { printf("WARNING: Latlong must be used with subgrid model. Aborting..."); exit(1); }

The guard is gated behind `verbose`, so a normal run proceeds. CalcT then divides by
Parptr->dx, which is read raw from the DEM header and never latlong-converted (input.cpp),
so dx is 0.000036 DEGREES instead of 4 metres and the timestep collapses to ~3e-6 s. The
run reports a healthy timestep, pins every core, and never advances. Nothing in the output
points at the par.

The 30 m path was always fine because templates/savannah.par.j2 emits both. Only
hand-built pars can drift, so this checks them.
"""
from pathlib import Path

# (flag, requires, why) -- combinations LISFLOOD accepts silently but computes wrongly.
PAIRED = [("latlong", "sgc_enable",
           "latlong needs the subgrid model; without it CalcT uses dx in degrees and the "
           "timestep collapses to microseconds (pars.cpp:976, input.cpp:1459)")]


def check_par(par_path):
    """Return a list of problem strings. Empty means the par looks sane."""
    flags, problems = set(), []
    if not Path(par_path).is_file():
        return [f"{par_path}: no such par file"]
    for ln in Path(par_path).read_text().splitlines():
        ln = ln.split("#")[0].strip()
        if ln:
            flags.add(ln.split()[0])
    for flag, needs, why in PAIRED:
        if flag in flags and needs not in flags:
            problems.append(f"'{flag}' present but '{needs}' missing: {why}")
    for ref in ("DEMfile", "bcifile", "bdyfile", "manningfile"):
        for ln in Path(par_path).read_text().splitlines():
            p = ln.split("#")[0].split()
            if len(p) >= 2 and p[0] == ref:
                f = Path(par_path).parent / p[1]
                if not f.exists():
                    problems.append(f"{ref} -> {p[1]} does not exist")
                elif ref == "bdyfile":
                    problems += check_bdy(f)
    # Numeric fields must parse as floats. An unsubstituted placeholder (tstart <T0>)
    # otherwise reaches LISFLOOD as "error reading decimal param" after the job starts.
    for ln in Path(par_path).read_text().splitlines():
        p = ln.split("#")[0].split()
        if len(p) >= 2 and p[0] in ("tstart", "sim_time", "saveint", "massint",
                                    "initial_tstep", "max_Froude"):
            try:
                float(p[1])
            except ValueError:
                problems.append(f"{p[0]} is not a number: {p[1]!r}")
    return problems


def check_bdy(bdy_path):
    """Return problem strings for a .bdy, reading it the way LISFLOOD does.

    `LoadTimeSeries` (input.cpp:1819) consumes exactly `count` rows after each block header,
    skipping only comment lines, and rejects any time <= the previous one. The previous time
    starts at -1 s, so a block whose first sample is negative fails on its first row. A gauge
    record placed on the model clock routinely starts before model time zero, and the run then
    exits before the first timestep with no output directory to inspect.
    """
    problems = []
    lines = Path(bdy_path).read_text().splitlines()
    name, i = Path(bdy_path).name, 1          # line 0 is the file comment
    while i < len(lines):
        block = lines[i].strip(); i += 1
        if not block:
            continue
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith("#")):
            i += 1
        if i >= len(lines):
            problems.append(f"{name}: block '{block}' has no header"); break
        try:
            n = int(lines[i].split()[0])
        except (ValueError, IndexError):
            problems.append(f"{name}: block '{block}' header unreadable: {lines[i]!r}"); break
        i += 1
        prev = -1.0
        for k in range(n):
            if i >= len(lines):
                problems.append(f"{name}: block '{block}' ends {n - k} rows short of its count")
                return problems
            tok = lines[i].split(); i += 1
            if len(tok) < 2:
                problems.append(f"{name}: block '{block}' row {k} has {len(tok)} token(s): "
                                f"{lines[i - 1]!r}")
                return problems
            t = float(tok[1])
            if t <= prev:
                why = ("first sample predates model time zero" if k == 0
                       else "times not strictly increasing")
                problems.append(f"{name}: block '{block}' row {k}: t={t} <= {prev} ({why})")
                return problems
            prev = t
    return problems


if __name__ == "__main__":
    import sys
    bad = False
    for p in sys.argv[1:]:
        probs = check_par(p)
        print(f"{'FAIL' if probs else 'OK  '}  {p}")
        for x in probs:
            print(f"        {x}")
        bad |= bool(probs)
    sys.exit(1 if bad else 0)
