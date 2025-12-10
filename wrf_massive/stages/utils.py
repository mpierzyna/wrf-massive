from __future__ import annotations
from typing import List

import pathlib
import logging
import subprocess

import jinja2


def render_template(template_path: str | pathlib.Path, **context) -> str:
    """Render a Jinja2 template with the given context."""
    template_path = pathlib.Path(template_path)
    template_str = template_path.read_text()
    template = jinja2.Template(template_str)
    return template.render(**context)


def load_wps_wrf_namelist_tmpl(template_path: str | pathlib.Path) -> jinja2.Template:
    """Load namelist.wps or namelist.input template and return as jinja2.Template instance.
    Filter `all_domains` will repeat template options automatically for all domains.
    """
    template_path = pathlib.Path(template_path)

    # Read namelist template to determine number of domains
    n_domains = int(get_namelist_value(namelist_path=template_path, field="max_dom"))

    def _all_domains(val: str) -> str:
        """Return a string with the same value for all domains"""
        try:
            return ",\t".join([str(val)] * n_domains) + ","
        except TypeError:
            return "???"

    # Set up template renderer,
    env = jinja2.Environment(
        keep_trailing_newline=True,  # trailing new line HAS to be kept, otherwise WRF will fail!
        undefined=jinja2.StrictUndefined,  # raise error if we forget to set a variable defined in a template
    )

    # Set filter with requested number of domains.
    # Use `x | all_domains` in template to repeat a value for all domains.
    env.filters["all_domains"] = _all_domains

    # Load template
    tmpl = env.from_string(template_path.read_text())
    return tmpl


def run_cmd_logged(
    cmd: List[str | pathlib.Path],
    logger: logging.Logger,
    cwd: pathlib.Path | None = None,
    msg: str = "",
):
    """Run a command in `cwd` and log errors"""

    # Make all paths absolute
    cmd = [str(c.absolute()) if isinstance(c, pathlib.Path) else c for c in cmd]

    # Run command
    logger.debug("-> Running " + " ".join(cmd) + f" in {cwd if cwd else '.'}...")
    p = subprocess.run(cmd, check=False, stderr=subprocess.PIPE, text=True, cwd=cwd)

    # Check for errors and log them
    if p.returncode != 0:
        logger.error(f"Error ({p.returncode}) occured while {msg}:\n{p.stderr}")
        raise RuntimeError(f"Error ({p.returncode}) occured while {msg}. Check log for details.")
    logger.info(f"-> Ran {cmd[1]} successfully.")


def get_namelist_value(namelist_path: str | pathlib.Path, field: str) -> str | List[str]:
    """Extract a field value from a Fortran namelist file."""
    namelist_path = pathlib.Path(namelist_path)
    for line in namelist_path.read_text().splitlines():
        if field in line:
            key, value = line.split("=")  # left is key, right is value
            value = value.strip()
            value = value[:-1] if value.endswith(",") else value  # remove trailing comma
            value = value.split(",")  # split in case of multiple values per domain
            value = [v.strip() for v in value]  # strip whitespace
            if len(value) == 1:
                return value[0]  # return single value as string
            return value
    raise KeyError(f"Couldn't find field {field} in {namelist_path}")
