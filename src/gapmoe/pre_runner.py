from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u


OptionValue = Union[str, int, float, bool]
CoordinateValue = Union[str, int, float]
OptionSequence = Sequence[OptionValue]
OptionMap = Mapping[str, Union[OptionValue, OptionSequence]]


@dataclass(frozen=True)
class SourceSelection:
    """Source weighting and extinction options for calc_rho_profile.

    PreRunner enables Genulens source weighting by default. If no explicit
    selection options are supplied, calc_rho_profile uses the same fallback
    source weighting as genulens.cpp.
    """

    enabled: bool = False
    i_mag: Optional[float] = None
    i_mag_error: Optional[float] = None
    i_mag_range: Optional[Tuple[float, float]] = None
    vi_color: Optional[float] = None
    vi_color_error: Optional[float] = None
    vi_color_range: Optional[Tuple[float, float]] = None
    ai_rc: Optional[float] = None
    evi_rc: Optional[float] = None
    dm_rc: Optional[float] = None
    hdust_pc: Optional[float] = None
    gamma_ds: Optional[float] = None

    def to_options(self) -> Dict[str, Union[OptionValue, Tuple[float, float]]]:
        options: Dict[str, Union[OptionValue, Tuple[float, float]]] = {}
        if self.enabled:
            options["SOURCE"] = 1

        if self.i_mag is not None:
            options["Is"] = self.i_mag
        if self.i_mag_error is not None:
            options["Iserr"] = self.i_mag_error
        if self.i_mag_range is not None:
            options["Isrange"] = self.i_mag_range

        if self.vi_color is not None:
            options["VIs"] = self.vi_color
        if self.vi_color_error is not None:
            options["VIserr"] = self.vi_color_error
        if self.vi_color_range is not None:
            options["VIsrange"] = self.vi_color_range

        if self.ai_rc is not None:
            options["AIrc"] = self.ai_rc
        if self.evi_rc is not None:
            options["EVIrc"] = self.evi_rc
        if self.dm_rc is not None:
            options["DMrc"] = self.dm_rc
        if self.hdust_pc is not None:
            options["hdust"] = self.hdust_pc
        if self.gamma_ds is not None:
            options["gammaDs"] = self.gamma_ds
        return options


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
    commands: Dict[str, Sequence[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class GenulensEnvironment:
    """Resolved Genulens pre_gapmoe environment status."""

    genulens_root: Path
    pre_gapmoe_dir: Path
    available_tools: Tuple[str, ...]
    missing_tools: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_tools


class PreRunner:
    """Run Genulens pre_gapmoe tools and write per-event GAPMOE inputs."""

    required_tools = ("calc_rho_profile", "calc_mass_dist", "calc_murel_dist")

    def __init__(
        self,
        genulens_root: Optional[Union[str, Path]] = None,
        output_dir: str | Path = ".",
        *,
        auto_build: bool = False,
    ) -> None:
        self.genulens_root = self._resolve_genulens_root(genulens_root)
        self.pre_gapmoe_dir = self.genulens_root / "pre_gapmoe"
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.auto_build = auto_build

    def check_environment(self) -> GenulensEnvironment:
        """Return the resolved Genulens pre_gapmoe executable status."""

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
        source: Optional[SourceSelection] = None,
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
        Genulens density precision; murel uses a coarser 250 pc distance grid.
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

        mass_cmd = self._command("calc_mass_dist", self._merge_options(base_options, mass_options))
        self._run_to_file(mass_cmd, mass_path)
        commands["mass"] = mass_cmd

        rho_base: Dict[str, Union[OptionValue, OptionSequence]] = {
            **base_options,
            "SOURCE": 1,
            "Dmin": d_min_pc,
            "Dmax": rho_max_pc,
            "Dstep": rho_d_step_pc,
        }
        if source is not None:
            rho_base.update(source.to_options())
        rho_cmd = self._command("calc_rho_profile", self._merge_options(rho_base, rho_options))
        self._run_to_file(rho_cmd, rho_path)
        commands["rho"] = rho_cmd

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
            "AUTOERR": int(autoerr),
        }
        if err_target is not None:
            murel_base["ERR_TARGET"] = err_target
        murel_cmd = self._command("calc_murel_dist", self._merge_options(murel_base, murel_options))
        self._run_to_file(murel_cmd, murel_path)
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
            "Could not find a Genulens checkout containing pre_gapmoe. "
            "Pass genulens_root=... or set GAPMOE_GENULENS_ROOT. "
            "The path may point either to the Genulens root or to pre_gapmoe itself. "
            f"Checked: {checked}"
        )

    def _prepare(self) -> None:
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
            "commands": result.commands,
        }
        result.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
