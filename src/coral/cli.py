"""coral CLI — thin wrapper that loads a scenario config and runs a stage.

Examples:
  coral render-par   configs/scenarios/savannah_matthew_LR.yaml runs/sav/savannah.par
  coral build-bdy    configs/scenarios/savannah_matthew_LR.yaml <geoclaw_output> <bci_in> <out_dir>
  coral show         configs/scenarios/savannah_matthew_LR.yaml
"""
from __future__ import annotations
import os
from pathlib import Path
import typer
from . import config
from .couple import build_bdy as _bdy

app = typer.Typer(add_completion=False, help="CORAL coastal-flood workflow CLI")


@app.command()
def show(scenario: str):
    """Print the resolved scenario config (base + overrides)."""
    c = config.load(scenario)
    typer.echo(f"{c.name}: bbox={c.domain.bbox} res={c.domain.res_m}m  "
               f"drag={c.geoclaw.drag} sea_level={c.geoclaw.sea_level}  "
               f"tstart={c.tstart:.0f} sim_time={c.sim_time:.0f}")


@app.command()
def render_par(scenario: str, out: str):
    """Render the LISFLOOD .par from templates/savannah.par.j2."""
    from jinja2 import Template
    import dataclasses
    c = config.load(scenario)
    # tstart/sim_time come from the config properties (single source of truth),
    # not recomputed in the template.
    ctx = {"name": c.name, "tstart": c.tstart, "sim_time": c.sim_time,
           "coupling": c.coupling, "lisflood": c.lisflood, "geoclaw": c.geoclaw,
           "forcing": c.forcing}
    # dataclasses -> attribute access works in jinja
    tpl = Template(Path("templates/savannah.par.j2").read_text())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(tpl.render(**ctx))
    typer.echo(f"wrote {out}")


@app.command()
def build_bdy(scenario: str, geoclaw_output: str, bci_in: str, out_dir: str,
              n_coupling: int = None, tide_file: str = None):
    """GeoClaw gauges -> .bdy + filtered .bci (filter/sort/shift/datum).

    tide_file overrides the series make_tide would fetch. Pass a constant series to run the
    static-tide arm of a tide comparison, so the two runs differ only in whether the tide
    varies in time.
    """
    c = config.load(scenario)
    os.makedirs(out_dir, exist_ok=True)
    res = _bdy.from_config(c, geoclaw_output, bci_in,
                           os.path.join(out_dir, f"{c.name}.bdy"),
                           os.path.join(out_dir, f"{c.name}_coastline.bci"),
                           n_coupling=n_coupling, tide_file=tide_file)
    typer.echo(f"kept {len(res['kept'])} gauges, dropped {len(res['dropped'])}, "
               f"time_offset +{res['time_offset']:.0f}s, bci {res['bci_points']}")


if __name__ == "__main__":
    app()
