from __future__ import annotations

import json
import importlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import numpy as np

from gapmoe.source_selection import CmdPriorTable, GenulensSourceModel


OptionValue = Union[str, int, float, bool]
CoordinateValue = Union[str, int, float]
OptionSequence = Sequence[OptionValue]
OptionMap = Mapping[str, Union[OptionValue, OptionSequence]]


@dataclass(frozen=True)
class PreRunResult:
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    l_deg: float
    b_deg: float
    output_dir: Path
    mass_path: Path
    rho_path: Path
    murel_path: Path
    manifest_path: Path
    source_evidence_path: Path | None = None
    cmd_prior_path: Path | None = None
    commands: Dict[str, Sequence[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class GenulensEnvironment:
    """Resolved genulens pre_gapmoe environment status."""

    genulens_root: Path
    pre_gapmoe_dir: Path
    available_tools: Tuple[str, ...]
    missing_tools: Tuple[str, ...]
    backend: str = "cli"

    @property
    def ok(self) -> bool:
        return not self.missing_tools


class PreRunner:
    """Run genulens pre_gapmoe tools and write per-event gapmoe inputs."""

    required_tools = ("calc_rho_profile", "calc_mass_dist", "calc_murel_dist")

    def __init__(
        self,
        genulens_root: Optional[Union[str, Path]] = None,
        output_dir: str | Path = ".",
        *,
        auto_build: bool = False,
        backend: Literal["auto", "python", "cli"] = "auto",
    ) -> None:
        if backend not in {"auto", "python", "cli"}:
            raise ValueError("backend must be 'auto', 'python', or 'cli'")
        self.backend = self._resolve_backend(backend, genulens_root)
        self.genulens_root = self._resolve_genulens_root(genulens_root) if self.backend == "cli" else None
        self.pre_gapmoe_dir = self.genulens_root / "pre_gapmoe" if self.genulens_root is not None else None
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.auto_build = auto_build

    def check_environment(self) -> GenulensEnvironment:
        """Return the resolved genulens pre_gapmoe executable status."""

        if self.backend == "python":
            pre_gapmoe = self._require_python_api()
            module_path = Path(getattr(importlib.import_module("genulens"), "__file__", ".")).resolve()
            available = [
                name for name in self.required_tools if hasattr(pre_gapmoe, self._python_api_name(name))
            ]
            missing = [name for name in self.required_tools if name not in available]
            return GenulensEnvironment(
                genulens_root=module_path,
                pre_gapmoe_dir=module_path.parent,
                available_tools=tuple(available),
                missing_tools=tuple(missing),
                backend="python",
            )

        assert self.pre_gapmoe_dir is not None
        assert self.genulens_root is not None
        available = []
        missing = []
        for tool in self.required_tools:
            if (self.pre_gapmoe_dir / tool).is_file():
                available.append(tool)
            else:
                missing.append(tool)
        return GenulensEnvironment(
            genulens_root=self.genulens_root,
            pre_gapmoe_dir=self.pre_gapmoe_dir,
            available_tools=tuple(available),
            missing_tools=tuple(missing),
            backend="cli",
        )

    def run(
        self,
        ra_deg: Optional[CoordinateValue] = None,
        dec_deg: Optional[CoordinateValue] = None,
        *,
        ra: Optional[CoordinateValue] = None,
        dec: Optional[CoordinateValue] = None,
        l_deg: Optional[CoordinateValue] = None,
        b_deg: Optional[CoordinateValue] = None,
        l: Optional[CoordinateValue] = None,
        b: Optional[CoordinateValue] = None,
        glon: Optional[CoordinateValue] = None,
        glat: Optional[CoordinateValue] = None,
        gal_l: Optional[CoordinateValue] = None,
        gal_b: Optional[CoordinateValue] = None,
        galactic_l: Optional[CoordinateValue] = None,
        galactic_b: Optional[CoordinateValue] = None,
        source_model: GenulensSourceModel | None = None,
        cmd_prior: CmdPriorTable | None = None,
        run_name: Optional[str] = None,
        distance_max_pc: float = 16000.0,
        rho_step_pc: float = 1.0,
        murel_distance_step_pc: float = 250.0,
        d_min_pc: float = 100.0,
        d_max_pc: Optional[float] = None,
        d_step_pc: Optional[float] = None,
        dl_min_pc: float = 0.0,
        dl_max_pc: Optional[float] = None,
        dl_step_pc: Optional[float] = None,
        ds_min_pc: float = 0.0,
        ds_max_pc: Optional[float] = None,
        ds_step_pc: Optional[float] = None,
        n_simu: int = 10_000_000,
        mu_max_masyr: float = 300.0,
        dmu_masyr: float = 0.5,
        autoerr: bool = True,
        err_target: Optional[float] = None,
        seed: Optional[int] = None,
        mass_options: Optional[Mapping[str, OptionValue]] = None,
        rho_options: Optional[OptionMap] = None,
        murel_options: Optional[Mapping[str, OptionValue]] = None,
        model_options: Optional[Mapping[str, OptionValue]] = None,
    ) -> PreRunResult:
        """Run mass, rho, and murel preprocessing for one sky position.

        The normal user-facing inputs are the sky coordinates and optional
        source-selection settings. By default, rho and murel preprocessing use
        the same maximum distance. Rho uses a 1 pc distance step to preserve the
        genulens density precision; murel uses a coarser 250 pc distance grid.
        The separate d/DL/DS options are advanced overrides.

        `calc_murel_dist` has no t0 or Earth-velocity option. Its output is the
        heliocentric relative proper-motion distribution from the Galactic
        lens/source kinematics.
        """

        self._prepare()
        ra_value, dec_value, l_value, b_value = self._resolve_coordinates(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            ra=ra,
            dec=dec,
            l_deg=l_deg,
            b_deg=b_deg,
            l=l,
            b=b,
            glon=glon,
            glat=glat,
            gal_l=gal_l,
            gal_b=gal_b,
            galactic_l=galactic_l,
            galactic_b=galactic_b,
        )

        label = run_name or self._default_run_name(l_value, b_value)
        run_dir = self.output_dir / label
        mass_path = run_dir / "mass.dat"
        rho_path = run_dir / "rho.dat"
        murel_path = run_dir / "murel.dat"
        manifest_path = run_dir / "manifest.json"
        source_evidence_path = run_dir / "source_evidence.npz"
        cmd_prior_path = run_dir / "cmd_prior.npz" if cmd_prior is not None else None

        for path in (mass_path, rho_path, murel_path, manifest_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        base_options: Dict[str, OptionValue] = {"l": l_value, "b": b_value}
        if seed is not None:
            base_options["seed"] = seed
        if model_options:
            base_options.update(model_options)

        rho_max_pc = distance_max_pc if d_max_pc is None else d_max_pc
        rho_d_step_pc = rho_step_pc if d_step_pc is None else d_step_pc
        murel_dl_max_pc = distance_max_pc if dl_max_pc is None else dl_max_pc
        murel_dl_step_pc = murel_distance_step_pc if dl_step_pc is None else dl_step_pc
        murel_ds_max_pc = distance_max_pc if ds_max_pc is None else ds_max_pc
        murel_ds_step_pc = murel_distance_step_pc if ds_step_pc is None else ds_step_pc

        commands: Dict[str, Sequence[str]] = {}

        mass_options_merged = self._merge_options(base_options, mass_options)
        mass_cmd = self._run_tool("calc_mass_dist", mass_options_merged, mass_path)
        commands["mass"] = mass_cmd

        rho_base: Dict[str, Union[OptionValue, OptionSequence]] = {
            **base_options,
            # gapmoe reconstructs the source base nMS * D_S^2 itself.  Do not
            # request calc_rho_profile's legacy gammaDs/rhoD_S output.
            "SOURCE": 0,
            "Dmin": d_min_pc,
            "Dmax": rho_max_pc,
            "Dstep": rho_d_step_pc,
        }
        rho_options_merged = self._merge_options(rho_base, rho_options)
        rho_cmd = self._run_tool("calc_rho_profile", rho_options_merged, rho_path)
        commands["rho"] = rho_cmd

        table = self._build_forward_source_table(
            source_model or GenulensSourceModel(),
            self._rho_distance_grid(rho_path),
            cmd_prior=cmd_prior,
        )
        table.save_npz(source_evidence_path)
        if cmd_prior_path is not None:
            cmd_prior.save_npz(cmd_prior_path)

        murel_base: Dict[str, OptionValue] = {
            **base_options,
            "GRID": 1,
            "DLmin": dl_min_pc,
            "DLmax": murel_dl_max_pc,
            "DLstep": murel_dl_step_pc,
            "DSmin": ds_min_pc,
            "DSmax": murel_ds_max_pc,
            "DSstep": murel_ds_step_pc,
            "Nsimu": n_simu,
            "mumax": mu_max_masyr,
            "dmu": dmu_masyr,
            "SOURCEGROUPS": 1,
            "AUTOERR": int(autoerr),
        }
        if err_target is not None:
            murel_base["ERR_TARGET"] = err_target
        murel_options_merged = self._merge_options(murel_base, murel_options)
        murel_cmd = self._run_tool("calc_murel_dist", murel_options_merged, murel_path)
        commands["murel"] = murel_cmd

        result = PreRunResult(
            ra_deg=ra_value,
            dec_deg=dec_value,
            l_deg=l_value,
            b_deg=b_value,
            output_dir=run_dir,
            mass_path=mass_path,
            rho_path=rho_path,
            murel_path=murel_path,
            manifest_path=manifest_path,
            source_evidence_path=source_evidence_path,
            cmd_prior_path=cmd_prior_path,
            commands=commands,
        )
        self._write_manifest(result)
        return result

    @classmethod
    def _resolve_genulens_root(cls, genulens_root: Optional[Union[str, Path]]) -> Path:
        candidates = []
        if genulens_root is not None:
            candidates.append(Path(genulens_root))
        env_root = os.environ.get("GAPMOE_GENULENS_ROOT") or os.environ.get("GENULENS_ROOT")
        if env_root:
            candidates.append(Path(env_root))
        candidates.extend([Path("../genulens"), Path.cwd() / "genulens"])

        for candidate in candidates:
            root = candidate.expanduser().resolve()
            if (root / "pre_gapmoe").is_dir():
                return root
            if root.name == "pre_gapmoe" and root.is_dir():
                return root.parent

        checked = ", ".join(str(path.expanduser()) for path in candidates)
        raise FileNotFoundError(
            "Could not find a genulens checkout containing pre_gapmoe. "
            "Pass genulens_root=... or set GAPMOE_GENULENS_ROOT. "
            "The path may point either to the genulens root or to pre_gapmoe itself. "
            f"Checked: {checked}"
        )

    @classmethod
    def _resolve_backend(cls, backend: str, genulens_root: Optional[Union[str, Path]]) -> str:
        if backend != "auto":
            return backend
        if genulens_root is not None:
            return "cli"
        return "python" if cls._python_api_available() else "cli"

    def _prepare(self) -> None:
        if self.backend == "python":
            missing = list(self.check_environment().missing_tools)
            if missing:
                names = ", ".join(missing)
                raise RuntimeError(f"Missing genulens.pre_gapmoe API function(s): {names}.")
            return

        assert self.pre_gapmoe_dir is not None
        assert self.genulens_root is not None
        if not self.pre_gapmoe_dir.is_dir():
            raise FileNotFoundError(f"pre_gapmoe directory not found: {self.pre_gapmoe_dir}")
        missing = list(self.check_environment().missing_tools)
        if missing and self.auto_build:
            subprocess.run(["make", "-C", str(self.pre_gapmoe_dir)], cwd=self.genulens_root, check=True)
            missing = list(self.check_environment().missing_tools)
        if missing:
            names = ", ".join(missing)
            raise FileNotFoundError(f"Missing pre_gapmoe executable(s): {names}. Build {self.pre_gapmoe_dir} first.")

    @staticmethod
    def _radec_to_lb(ra_deg: float, dec_deg: float) -> Tuple[float, float]:
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        l_deg = coord.galactic.l.deg
        if l_deg > 180.0:
            l_deg -= 360.0
        return l_deg, coord.galactic.b.deg

    @classmethod
    def _resolve_coordinates(
        cls,
        *,
        ra_deg: Optional[CoordinateValue] = None,
        dec_deg: Optional[CoordinateValue] = None,
        ra: Optional[CoordinateValue] = None,
        dec: Optional[CoordinateValue] = None,
        l_deg: Optional[CoordinateValue] = None,
        b_deg: Optional[CoordinateValue] = None,
        l: Optional[CoordinateValue] = None,
        b: Optional[CoordinateValue] = None,
        glon: Optional[CoordinateValue] = None,
        glat: Optional[CoordinateValue] = None,
        gal_l: Optional[CoordinateValue] = None,
        gal_b: Optional[CoordinateValue] = None,
        galactic_l: Optional[CoordinateValue] = None,
        galactic_b: Optional[CoordinateValue] = None,
    ) -> Tuple[Optional[float], Optional[float], float, float]:
        ra_value = cls._pick_coordinate("ra", {"ra_deg": ra_deg, "ra": ra}, cls._parse_ra)
        dec_value = cls._pick_coordinate("dec", {"dec_deg": dec_deg, "dec": dec}, cls._parse_degree_angle)
        l_value = cls._pick_coordinate(
            "galactic longitude",
            {"l_deg": l_deg, "l": l, "glon": glon, "gal_l": gal_l, "galactic_l": galactic_l},
            cls._parse_degree_angle,
        )
        b_value = cls._pick_coordinate(
            "galactic latitude",
            {"b_deg": b_deg, "b": b, "glat": glat, "gal_b": gal_b, "galactic_b": galactic_b},
            cls._parse_degree_angle,
        )

        has_equatorial = ra_value is not None or dec_value is not None
        has_galactic = l_value is not None or b_value is not None

        if has_equatorial and has_galactic:
            raise ValueError("Specify either RA/Dec or Galactic l/b, not both.")
        if has_equatorial:
            if ra_value is None or dec_value is None:
                raise ValueError("Both RA and Dec are required. Accepted names: ra_deg/dec_deg or ra/dec.")
            l_calc, b_calc = cls._radec_to_lb(ra_value, dec_value)
            return ra_value, dec_value, l_calc, b_calc
        if has_galactic:
            if l_value is None or b_value is None:
                raise ValueError(
                    "Both Galactic longitude and latitude are required. "
                    "Accepted names: l_deg/b_deg, l/b, glon/glat, gal_l/gal_b, galactic_l/galactic_b."
                )
            if l_value > 180.0:
                l_value -= 360.0
            return None, None, l_value, b_value
        raise ValueError(
            "Sky position is required. Use RA/Dec names ra_deg/dec_deg or ra/dec, "
            "or Galactic names l_deg/b_deg, l/b, glon/glat, gal_l/gal_b, galactic_l/galactic_b."
        )

    @staticmethod
    def _pick_coordinate(
        name: str,
        values: Mapping[str, Optional[CoordinateValue]],
        parser,
    ) -> Optional[float]:
        supplied = [(key, value) for key, value in values.items() if value is not None]
        if not supplied:
            return None
        first_key, first_raw = supplied[0]
        first_value = parser(first_raw)
        conflicts = []
        for key, raw_value in supplied[1:]:
            parsed_value = parser(raw_value)
            if abs(parsed_value - first_value) > 1e-10:
                conflicts.append(key)
        if conflicts:
            names = ", ".join([first_key] + conflicts)
            raise ValueError(f"Conflicting values supplied for {name}: {names}")
        return first_value

    @staticmethod
    def _parse_ra(value: CoordinateValue) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        lower = text.lower()
        if ":" in text or "h" in lower:
            return Angle(text, unit=u.hourangle).to_value(u.deg)
        return Angle(text, unit=u.deg).to_value(u.deg)

    @staticmethod
    def _parse_degree_angle(value: CoordinateValue) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return Angle(str(value).strip(), unit=u.deg).to_value(u.deg)

    @staticmethod
    def _default_run_name(l_deg: float, b_deg: float) -> str:
        return f"lb_{l_deg:+.6f}_{b_deg:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")

    def _command(
        self,
        tool: str,
        options: Mapping[str, Optional[Union[OptionValue, OptionSequence]]],
    ) -> Sequence[str]:
        cmd = [str(Path("pre_gapmoe") / tool)]
        for key, value in options.items():
            if value is None:
                continue
            cmd.append(str(key))
            if isinstance(value, bool):
                cmd.append("1" if value else "0")
            elif isinstance(value, (str, int, float)):
                cmd.append(str(value))
            else:
                cmd.extend(str(item) for item in value)
        return cmd

    def _run_tool(
        self,
        tool: str,
        options: Mapping[str, Optional[Union[OptionValue, OptionSequence]]],
        output_path: Path,
    ) -> Sequence[str]:
        cmd = self._command(tool, options)
        if self.backend == "python":
            pre_gapmoe = self._require_python_api()
            function = getattr(pre_gapmoe, self._python_api_name(tool))
            kwargs = {key: value for key, value in options.items() if value is not None}
            table = function(**kwargs)
            output_path.write_text(table.stdout)
        else:
            self._run_to_file(cmd, output_path)
        return cmd

    @staticmethod
    def _merge_options(
        base: OptionMap,
        override: Optional[OptionMap],
    ) -> Dict[str, Union[OptionValue, OptionSequence]]:
        merged = dict(base)
        if override:
            merged.update(override)
        return merged

    def _run_to_file(self, cmd: Sequence[str], output_path: Path) -> None:
        assert self.genulens_root is not None
        with output_path.open("w") as stdout:
            subprocess.run(cmd, cwd=self.genulens_root, stdout=stdout, check=True)

    def _write_manifest(self, result: PreRunResult) -> None:
        payload = {
            "ra_deg": result.ra_deg,
            "dec_deg": result.dec_deg,
            "l_deg": result.l_deg,
            "b_deg": result.b_deg,
            "genulens_root": str(self.genulens_root),
            "output_dir": str(result.output_dir),
            "mass_path": str(result.mass_path),
            "rho_path": str(result.rho_path),
            "murel_path": str(result.murel_path),
            "source_evidence_path": (
                str(result.source_evidence_path) if result.source_evidence_path is not None else None
            ),
            "cmd_prior_path": str(result.cmd_prior_path) if result.cmd_prior_path is not None else None,
            "commands": result.commands,
        }
        result.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    @staticmethod
    def _rho_distance_grid(path: Path) -> np.ndarray:
        data = np.atleast_2d(np.genfromtxt(path, comments="#"))
        if data.size == 0 or data.shape[1] < 1:
            raise ValueError(f"rho profile has no distance grid: {path}")
        return np.asarray(data[:, 0], dtype=float)

    def _build_forward_source_table(
        self,
        source_model: GenulensSourceModel,
        distance_pc: np.ndarray,
        *,
        cmd_prior: CmdPriorTable | None = None,
    ):
        if self.backend != "cli" or self.genulens_root is None:
            return source_model.build_evidence_grid(distance_pc, cmd_prior=cmd_prior)
        previous = os.environ.get("GENULENS_INPUT_DIR")
        os.environ["GENULENS_INPUT_DIR"] = str(self.genulens_root / "input_files")
        try:
            return source_model.build_evidence_grid(distance_pc, cmd_prior=cmd_prior)
        finally:
            if previous is None:
                os.environ.pop("GENULENS_INPUT_DIR", None)
            else:
                os.environ["GENULENS_INPUT_DIR"] = previous

    @staticmethod
    def _python_api_name(tool: str) -> str:
        names = {
            "calc_mass_dist": "mass_distribution",
            "calc_rho_profile": "rho_profile",
            "calc_murel_dist": "murel_distribution",
        }
        return names[tool]

    @classmethod
    def _python_api_available(cls) -> bool:
        try:
            pre_gapmoe = importlib.import_module("genulens").pre_gapmoe
        except Exception:
            return False
        return all(hasattr(pre_gapmoe, cls._python_api_name(tool)) for tool in cls.required_tools)

    @staticmethod
    def _require_python_api() -> Any:
        try:
            return importlib.import_module("genulens").pre_gapmoe
        except Exception as exc:
            raise RuntimeError(
                "genulens.pre_gapmoe is not available. Install a genulens wheel with the pre_gapmoe Python API "
                "or use backend='cli' with a local genulens checkout."
            ) from exc
