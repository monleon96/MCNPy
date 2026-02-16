"""
NJOY Input Deck builder.

Provides the fluent ``InputDeck`` class for notebook/scripting use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .modules import GENERATORS, generate_module
from .viewr_defs import ViewrPlot
from .plotr_defs import PlotrJob


class InputDeck:
    """Builder for NJOY input decks — add modules, then render or save.

    Usage::

        deck = InputDeck()
        deck.moder(nin=20, nout=-25)
        deck.reconr(isotope="U235", err=0.001)
        deck.broadr(isotope="U235", temperatures=[293.6])
        deck.acer(isotope="U235", tempd=293.6, suff=0.03)
        print(deck)
        deck.save("u235.inp")
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._modules: list[str] = []
        # Auto-chaining state
        self._endf_tape: int | None = None
        self._pendf_tape: int | None = None
        self._gendf_tape: int | None = None
        self._leapr_tape: int | None = None
        self._plotr_tape: int | None = None
        self._next_tape: int = -20

    # ------------------------------------------------------------------
    # Auto-chaining helpers
    # ------------------------------------------------------------------

    def _alloc_tape(self) -> int:
        """Allocate the next available tape number (negative = binary)."""
        self._next_tape -= 1
        return self._next_tape

    def _resolve_input(self, explicit: int | None, attr: str, label: str) -> int:
        """Resolve an input tape: explicit value > tracked state > error."""
        if explicit is not None:
            return explicit
        tracked = getattr(self, attr)
        if tracked is not None:
            return tracked
        raise ValueError(
            f"No {label} tape available. Call the preceding module first "
            f"or pass the tape number explicitly."
        )

    # ------------------------------------------------------------------
    # Module methods — each maps friendly kwargs → internal param dict
    # ------------------------------------------------------------------

    def moder(self, *, nin: int = 20, nout: int | None = None) -> InputDeck:
        """Add a MODER step."""
        nout = nout if nout is not None else self._alloc_tape()
        self._append("MODER", {"nin": nin, "nout": nout})
        self._endf_tape = nout
        return self

    def reconr(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        isotope: str = "U235",
        err: float = 0.001,
        tempr: float | None = None,
        errmax: float | None = None,
        errint: float | None = None,
    ) -> InputDeck:
        """Add a RECONR step."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        npend = npend if npend is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "mat": isotope,
            "err": str(err),
        }
        if tempr is not None:
            p["tempr"] = tempr
        if errmax is not None:
            p["errmax"] = errmax
        if errint is not None:
            p["errint"] = errint
        self._append("RECONR", p)
        self._pendf_tape = npend
        return self

    def broadr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        isotope: str = "U235",
        temperatures: list[float] | str = "293.6",
        errthn: float = 0.001,
        thnmax: float | None = None,
        errmax: float | None = None,
        errint: float | None = None,
        istart: str | None = None,
        istrap: str | None = None,
        temp1: float | None = None,
    ) -> InputDeck:
        """Add a BROADR step."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "mat": isotope,
            "temp2": temp_str,
            "errthn": str(errthn),
        }
        if thnmax is not None:
            p["thnmax"] = thnmax
        if errmax is not None:
            p["errmax"] = errmax
        if errint is not None:
            p["errint"] = errint
        if istart is not None:
            p["istart"] = istart
        if istrap is not None:
            p["istrap"] = istrap
        if temp1 is not None:
            p["temp1"] = temp1
        self._append("BROADR", p)
        self._pendf_tape = nout
        return self

    def heatr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        nplot: int = 0,
        isotope: str = "U235",
        mtk: list[int] | str = "302",
        ntemp: int = 0,
        local: str | None = None,
        iprint: str | None = None,
        ed: float | None = None,
    ) -> InputDeck:
        """Add a HEATR step."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(mtk, list):
            mtk_str = " ".join(str(m) for m in mtk)
        else:
            mtk_str = str(mtk)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "nplot": str(nplot),
            "matd": isotope,
            "mtk": mtk_str,
        }
        if ntemp:
            p["ntemp"] = ntemp
        if local is not None:
            p["local"] = local
        if iprint is not None:
            p["iprint"] = iprint
        if ed is not None:
            p["ed"] = ed
        self._append("HEATR", p)
        self._pendf_tape = nout
        return self

    def thermr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        matde: int = 0,
        isotope: str = "U235",
        nbin: int = 8,
        iin: int = 2,
        icoh: int = 0,
        iform: int | None = None,
        natom: int | None = None,
        mtref: int | None = None,
        temperatures: list[float] | str = "293.6",
        tol: float = 0.01,
        emax: float = 4.0,
        iprint: str | None = None,
    ) -> InputDeck:
        """Add a THERMR step (thermal scattering cross sections)."""
        # nendf is the thermal/LEAPR tape; 0 means free-gas
        if nendf is None:
            if matde == 0:
                nendf = 0
            else:
                nendf = self._resolve_input(None, '_leapr_tape', 'LEAPR/thermal')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "matde": matde,
            "matdp": isotope,
            "nbin": nbin,
            "iin": iin,
            "icoh": icoh,
            "temperatures": temp_str,
            "tol": tol,
            "emax": emax,
        }
        if iform is not None:
            p["iform"] = iform
        if natom is not None:
            p["natom"] = natom
        if mtref is not None:
            p["mtref"] = mtref
        if iprint is not None:
            p["iprint"] = iprint
        self._append("THERMR", p)
        self._pendf_tape = nout
        return self

    def purr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        isotope: str = "U235",
        temperatures: list[float] | str = "293.6",
        sigz: list[float] | str = "1e10",
        nbin: int = 20,
        nladr: int = 64,
        iprint: str | None = None,
        nunx: int | None = None,
    ) -> InputDeck:
        """Add a PURR step."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        if isinstance(sigz, list):
            sigz_str = " ".join(str(s) for s in sigz)
        else:
            sigz_str = str(sigz)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "matd": isotope,
            "temp": temp_str,
            "sigz": sigz_str,
            "nbin": str(nbin),
            "nladr": str(nladr),
        }
        if iprint is not None:
            p["iprint"] = iprint
        if nunx is not None:
            p["nunx"] = nunx
        self._append("PURR", p)
        self._pendf_tape = nout
        return self

    def unresr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        isotope: str = "U235",
        temperatures: list[float] | str = "293.6",
        sigz: list[float] | str = "1e10",
        iprint: str | None = None,
    ) -> InputDeck:
        """Add an UNRESR step."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        if isinstance(sigz, list):
            sigz_str = " ".join(str(s) for s in sigz)
        else:
            sigz_str = str(sigz)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "matd": isotope,
            "temp": temp_str,
            "sigz": sigz_str,
        }
        if iprint is not None:
            p["iprint"] = iprint
        self._append("UNRESR", p)
        self._pendf_tape = nout
        return self

    def gaspr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
    ) -> InputDeck:
        """Add a GASPR step."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        self._append("GASPR", {"nendf": nendf, "nin": nin, "nout": nout})
        self._pendf_tape = nout
        return self

    def groupr(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        ngout1: int = 0,
        ngout2: int | None = None,
        isotope: str = "U235",
        ign: int = 3,
        igg: int = 0,
        iwt: int = 3,
        lord: int = 0,
        temperatures: list[float] | str = "293.6",
        sigz: list[float] | str = "1e10",
        title: str | None = None,
        reactions: list[tuple] | None = None,
        egn: list[float] | None = None,
        egg: list[float] | None = None,
        eb: float | None = None,
        tb: float | None = None,
        ec: float | None = None,
        tc: float | None = None,
        iprint: str | None = None,
        ismooth: int | None = None,
    ) -> InputDeck:
        """Add a GROUPR step."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        npend = self._resolve_input(npend, '_pendf_tape', 'PENDF')
        ngout2 = ngout2 if ngout2 is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        if isinstance(sigz, list):
            sigz_str = " ".join(str(s) for s in sigz)
        else:
            sigz_str = str(sigz)
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "ngout1": ngout1,
            "ngout2": ngout2,
            "matb": isotope,
            "ign": ign,
            "igg": igg,
            "iwt": iwt,
            "lord": lord,
            "temperatures": temp_str,
            "sigz": sigz_str,
        }
        if title is not None:
            p["title"] = title
        if reactions is not None:
            p["reactions"] = reactions
        if egn is not None:
            p["egn"] = egn
        if egg is not None:
            p["egg"] = egg
        if eb is not None:
            p["eb"] = eb
        if tb is not None:
            p["tb"] = tb
        if ec is not None:
            p["ec"] = ec
        if tc is not None:
            p["tc"] = tc
        if iprint is not None:
            p["iprint"] = iprint
        if ismooth is not None:
            p["ismooth"] = ismooth
        self._append("GROUPR", p)
        self._gendf_tape = ngout2
        return self

    def gaminr(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        ngam1: int = 0,
        ngam2: int | None = None,
        isotope: str = "U235",
        igg: int = 7,
        iwt: int = 3,
        lord: int = 0,
        title: str | None = None,
        reactions: list[tuple] | None = None,
        egg: list[float] | None = None,
        iprint: str | None = None,
    ) -> InputDeck:
        """Add a GAMINR step (multigroup photoatomic cross sections)."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        npend = self._resolve_input(npend, '_pendf_tape', 'PENDF')
        ngam2 = ngam2 if ngam2 is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "ngam1": ngam1,
            "ngam2": ngam2,
            "matb": isotope,
            "igg": igg,
            "iwt": iwt,
            "lord": lord,
        }
        if title is not None:
            p["title"] = title
        if reactions is not None:
            p["reactions"] = reactions
        if egg is not None:
            p["egg"] = egg
        if iprint is not None:
            p["iprint"] = iprint
        self._append("GAMINR", p)
        self._gendf_tape = ngam2
        return self

    def errorr(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        ngout: int = 0,
        nout: int | None = None,
        isotope: str = "U235",
        ign: int = 1,
        iwt: int = 6,
        iprint: str | None = None,
        irelco: int = 1,
        mprint: int = 0,
        tempin: float = 300.0,
        mfcov: int = 33,
        irespr: int | None = None,
        dap: float | None = None,
        egn: list[float] | None = None,
    ) -> InputDeck:
        """Add an ERRORR step."""
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        npend = self._resolve_input(npend, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "ngout": ngout,
            "nout": nout,
            "matd": isotope,
            "ign": ign,
            "iwt": iwt,
            "irelco": irelco,
            "mprint": mprint,
            "tempin": tempin,
            "mfcov": mfcov,
        }
        if iprint is not None:
            p["iprint"] = iprint
        if irespr is not None:
            p["irespr"] = irespr
        if dap is not None:
            p["dap"] = dap
        if egn is not None:
            p["egn"] = egn
        self._append("ERRORR", p)
        return self

    def acer(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        ngend: int = 0,
        nace: int = 40,
        ndir: int = 41,
        isotope: str = "U235",
        tempd: float = 293.6,
        suff: float = 0.03,
        iopt: int = 1,
        itype: int = 1,
        iprint: str | None = None,
        # Fast (iopt=1) params:
        newfor: int | None = None,
        iopp: int | None = None,
        ismooth: int | None = None,
        thin1: float | None = None,
        thin2: float | None = None,
        thin3: float | None = None,
        # Thermal (iopt=2) params:
        tname: str | None = None,
        iza: list[int] | str | None = None,
        mti: int | None = None,
        nbint: int = 16,
        mte: int = 0,
        ielas: int = 0,
        nmix: int = 1,
        emax: float = 1000.0,
        iwt: int = 2,
        # Extra iz,aw pairs:
        nxtra: list[tuple[int, float]] | None = None,
    ) -> InputDeck:
        """Add an ACER step."""
        abs_iopt = abs(iopt)
        # iopt 7/8 read existing ACE files — no ENDF/PENDF needed
        if abs_iopt in (7, 8):
            if nendf is None:
                nendf = 0
            if npend is None:
                npend = 0
        else:
            nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
            npend = self._resolve_input(npend, '_pendf_tape', 'PENDF')
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "ngend": ngend,
            "nace": nace,
            "ndir": ndir,
            "matd": isotope,
            "tempd": tempd,
            "suff": suff,
            "iopt": str(iopt),
            "itype": itype,
        }
        if iprint is not None:
            p["iprint"] = iprint
        # Fast params (iopt=1)
        if newfor is not None:
            p["newfor"] = newfor
        if iopp is not None:
            p["iopp"] = iopp
        if ismooth is not None:
            p["ismooth"] = ismooth
        if thin1 is not None:
            p["thin1"] = thin1
        if thin2 is not None:
            p["thin2"] = thin2
        if thin3 is not None:
            p["thin3"] = thin3
        # Thermal params — only add when provided or iopt=2
        if tname is not None:
            p["tname"] = tname
        if iza is not None:
            p["iza"] = iza
        if mti is not None:
            p["mti"] = mti
        if nbint != 16:
            p["nbint"] = nbint
        if mte != 0:
            p["mte"] = mte
        if ielas != 0:
            p["ielas"] = ielas
        if nmix != 1:
            p["nmix"] = nmix
        if emax != 1000.0:
            p["emax"] = emax
        if iwt != 2:
            p["iwt"] = iwt
        # Extra iz,aw pairs
        if nxtra is not None:
            p["nxtra"] = nxtra
        self._append("ACER", p)
        return self

    def covr(
        self,
        *,
        nin: int,
        nout: int = 0,
        nplot: int = 0,
        cases: list[list[int]] | None = None,
        # Plot mode params
        icolor: int = 0,
        tlev: list[float] | None = None,
        epmin: float | None = None,
        irelco: int = 1,
        noleg: int = 0,
        nstart: int = 1,
        ndiv: int = 1,
        # Library mode params
        matype: int = 3,
        hlibid: str = "",
        hdescr: str = "",
    ) -> InputDeck:
        """Add a COVR step (post-process ERRORR covariance data)."""
        p: dict[str, Any] = {"nin": nin, "nout": nout}
        if cases is not None:
            p["cases"] = cases
        if nout > 0:
            # Library mode
            p["matype"] = matype
            p["hlibid"] = hlibid
            p["hdescr"] = hdescr
        else:
            # Plot mode
            p["nplot"] = nplot
            p["icolor"] = icolor
            if tlev is not None:
                p["tlev"] = tlev
            if epmin is not None:
                p["epmin"] = epmin
            p["irelco"] = irelco
            p["noleg"] = noleg
            p["nstart"] = nstart
            p["ndiv"] = ndiv
        self._append("COVR", p)
        return self

    def leapr(
        self,
        *,
        mat: int,
        za: int,
        awr: float,
        spr: float,
        alphas: list[float],
        betas: list[float],
        temperatures: list[dict],
        nout: int | None = None,
        title: str = "thermal scattering law via leapr",
        npr: int = 1,
        iel: int = 0,
        ncold: int = 0,
        nsk: int = 0,
        nss: int = 0,
        b7: int = 0,
        aws: float = 0.0,
        sps: float = 0.0,
        mss: int = 0,
        lat: int = 1,
        isabt: int = 0,
        ilog: int = 0,
        smin: float | None = None,
        nphon: int = 100,
        iprint: str | None = None,
        secondary_temperatures: list[dict] | None = None,
        comments: list[str] | None = None,
    ) -> InputDeck:
        """Add a LEAPR step (thermal scattering law)."""
        nout = nout if nout is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "nout": nout,
            "title": title,
            "mat": mat,
            "za": za,
            "awr": awr,
            "spr": spr,
            "alphas": alphas,
            "betas": betas,
            "temperatures": temperatures,
            "npr": npr,
            "iel": iel,
            "ncold": ncold,
            "nsk": nsk,
            "nss": nss,
            "b7": b7,
            "aws": aws,
            "sps": sps,
            "mss": mss,
            "lat": lat,
            "isabt": isabt,
            "ilog": ilog,
            "nphon": nphon,
        }
        if smin is not None:
            p["smin"] = smin
        if iprint is not None:
            p["iprint"] = iprint
        if secondary_temperatures is not None:
            p["secondary_temperatures"] = secondary_temperatures
        if comments is not None:
            p["comments"] = comments
        self._append("LEAPR", p)
        self._leapr_tape = nout
        return self

    def resxsr(
        self,
        *,
        nout: int | None = None,
        materials: list[dict],
        maxt: int = 1,
        efirst: float = 4.0,
        elast: float = 500.0,
        eps: float = 0.001,
        huse: str = "NJOY",
        ivers: int = 0,
        comments: list[str] | None = None,
    ) -> InputDeck:
        """Add a RESXSR step (resonance cross section file).

        Each entry in *materials* is a dict with keys ``isotope`` (resolved
        to both the hollerith name and MAT number) and ``unit`` (PENDF tape
        number for that material).
        """
        nout = nout if nout is not None else self._alloc_tape()
        mat_list: list[dict] = []
        for m in materials:
            from .isotopes import get_mat_number
            isotope = m["isotope"]
            mat_list.append({
                "name": isotope,
                "mat": get_mat_number(isotope),
                "unit": m["unit"],
            })
        p: dict[str, Any] = {
            "nout": nout,
            "materials": mat_list,
            "maxt": maxt,
            "efirst": efirst,
            "elast": elast,
            "eps": eps,
            "huse": huse,
            "ivers": ivers,
        }
        if comments is not None:
            p["comments"] = comments
        self._append("RESXSR", p)
        return self

    def ccccr(
        self,
        *,
        nin: int | None = None,
        nisot: int = 50,
        nbrks: int = 0,
        ndlay: int = 0,
        isotopes: list[dict],
        ngroup: int,
        maxord: int = 0,
        ifopt: int = 1,
        lprint: int = 0,
        ivers: int = 0,
        huse: str = "",
        hsetid: str = "",
        nggrup: int = 0,
        # ISOTXS params
        nsblok: int = 1,
        maxup: int = 0,
        maxdn: int | None = None,
        ichix: int = -1,
        spec: list[float] | None = None,
        isotxs_params: list[dict] | None = None,
        # BRKOXS params
        nti: int = -1,
        nzi: int = -1,
        atem: list[float] | None = None,
        asig: list[float] | None = None,
    ) -> InputDeck:
        """Add a CCCCR step (CCCC interface files from GENDF)."""
        nin = self._resolve_input(nin, '_gendf_tape', 'GENDF')
        p: dict[str, Any] = {
            "nin": nin,
            "nisot": nisot,
            "nbrks": nbrks,
            "ndlay": ndlay,
            "isotopes": isotopes,
            "ngroup": ngroup,
            "nggrup": nggrup,
            "maxord": maxord,
            "ifopt": ifopt,
            "lprint": lprint,
            "ivers": ivers,
            "huse": huse,
            "hsetid": hsetid,
        }
        if nisot > 0:
            p["nsblok"] = nsblok
            p["maxup"] = maxup
            p["maxdn"] = maxdn if maxdn is not None else ngroup
            p["ichix"] = ichix
            if spec is not None:
                p["spec"] = spec
            if isotxs_params is not None:
                p["isotxs_params"] = isotxs_params
            else:
                p["isotxs_params"] = []
        if nbrks > 0:
            p["nti"] = nti
            p["nzi"] = nzi
            if atem is not None:
                p["atem"] = atem
            if asig is not None:
                p["asig"] = asig
        self._append("CCCCR", p)
        return self

    def matxsr(
        self,
        *,
        ngen1: int | None = None,
        ngen2: int = 0,
        nmatx: int | None = None,
        ivers: int = 0,
        huse: str = "",
        hsetid: list[str] | None = None,
        particles: list[dict] | None = None,
        data_types: list[dict] | None = None,
        materials: list[dict] | None = None,
        ngen3: int = 0, ngen4: int = 0, ngen5: int = 0,
        ngen6: int = 0, ngen7: int = 0, ngen8: int = 0,
    ) -> InputDeck:
        """Add a MATXSR step (MATXS library from GENDF tapes)."""
        ngen1 = self._resolve_input(ngen1, '_gendf_tape', 'GENDF')
        nmatx = nmatx if nmatx is not None else self._alloc_tape()
        if particles is None:
            raise ValueError("matxsr() requires 'particles'")
        if data_types is None:
            raise ValueError("matxsr() requires 'data_types'")
        if materials is None:
            raise ValueError("matxsr() requires 'materials'")
        if hsetid is None:
            hsetid = ["matxs library"]
        p: dict[str, Any] = {
            "ngen1": ngen1,
            "ngen2": ngen2,
            "nmatx": nmatx,
            "ivers": ivers,
            "huse": huse,
            "hsetid": hsetid,
            "particles": particles,
            "data_types": data_types,
            "materials": materials,
            "ngen3": ngen3,
            "ngen4": ngen4,
            "ngen5": ngen5,
            "ngen6": ngen6,
            "ngen7": ngen7,
            "ngen8": ngen8,
        }
        self._append("MATXSR", p)
        return self

    def wimsr(
        self,
        *,
        ngendf: int | None = None,
        nout: int | None = None,
        isotope: str = "U235",
        rdfid: float | None = None,
        iverw: int = 4,
        igroup: int = 0,
        # Card 4 options
        ntemp: int = 0,
        nsigz: int = 0,
        sgref: float = 1e10,
        ires: int = 0,
        sigp: float = 0.0,
        mti: int = 0,
        mtc: int = 0,
        ip1opt: int = 1,
        inorf: int = 0,
        isof: int = 0,
        ifprod: int = 0,
        jp1: int = 0,
        # Lambdas
        lambdas: list[float] | None = None,
        # Optional
        iprint: str | None = None,
        burnup: dict | None = None,
        p1flx: list[float] | None = None,
        # Custom groups (igroup=9)
        ngnd: int | None = None,
        nfg: int | None = None,
        nrg: int | None = None,
        igref: int | None = None,
    ) -> InputDeck:
        """Add a WIMSR step (WIMS library from GENDF tape)."""
        ngendf = self._resolve_input(ngendf, '_gendf_tape', 'GENDF')
        nout = nout if nout is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "ngendf": ngendf,
            "nout": nout,
            "mat": isotope,
            "iverw": iverw,
            "igroup": igroup,
            "ntemp": ntemp,
            "nsigz": nsigz,
            "sgref": sgref,
            "ires": ires,
            "sigp": sigp,
            "mti": mti,
            "mtc": mtc,
            "ip1opt": ip1opt,
            "inorf": inorf,
            "isof": isof,
            "ifprod": ifprod,
            "jp1": jp1,
        }
        if rdfid is not None:
            p["rdfid"] = rdfid
        if lambdas is not None:
            p["lambdas"] = lambdas
        if iprint is not None:
            p["iprint"] = iprint
        if burnup is not None:
            p["iburn"] = 1
            p["burnup"] = burnup
        if p1flx is not None:
            p["p1flx"] = p1flx
        if igroup == 9:
            if ngnd is not None:
                p["ngnd"] = ngnd
            if nfg is not None:
                p["nfg"] = nfg
            if nrg is not None:
                p["nrg"] = nrg
            if igref is not None:
                p["igref"] = igref
        self._append("WIMSR", p)
        return self

    def dtfr(
        self,
        *,
        nin: int | None = None,
        nout: int | None = None,
        npend: int | None = None,
        nplot: int = 0,
        iedit: int = 0,
        nlmax: int = 1,
        ng: int = 30,
        # DTF table params (iedit=0)
        iptotl: int | None = None,
        ipingp: int | None = None,
        itabl: int | None = None,
        edit_names: list[str] | None = None,
        edits: list[list[int]] | None = None,
        ntherm: int = 0,
        mti: int | None = None,
        mtc: int = 0,
        nlc: int = 0,
        # Gamma
        nptabl: int = 0,
        ngp: int = 0,
        # Materials
        materials: list[dict] | None = None,
        # Options
        iprint: str | None = None,
        ifilm: int = 0,
    ) -> InputDeck:
        """Add a DTFR step (DTF-IV transport tables from GENDF data)."""
        nin = self._resolve_input(nin, '_gendf_tape', 'GENDF')
        # DTF output is coded (ASCII) text — use positive tape number
        if nout is None:
            nout = abs(self._alloc_tape())
        # Optionally use PENDF tape for point-data plots
        if npend is None:
            npend = self._pendf_tape or 0
        p: dict[str, Any] = {
            "nin": nin,
            "nout": nout,
            "npend": npend,
            "nplot": nplot,
            "iedit": iedit,
            "nlmax": nlmax,
            "ng": ng,
            "nptabl": nptabl,
            "ngp": ngp,
            "ifilm": ifilm,
        }
        if iprint is not None:
            p["iprint"] = iprint
        if iedit == 0:
            if iptotl is not None:
                p["iptotl"] = iptotl
            if edit_names is not None:
                p["edit_names"] = edit_names
            if edits is not None:
                p["edits"] = edits
            if ipingp is not None:
                p["ipingp"] = ipingp
            if itabl is not None:
                p["itabl"] = itabl
            p["ntherm"] = ntherm
            if ntherm > 0:
                if mti is None:
                    raise ValueError("dtfr() with ntherm>0 requires 'mti'")
                p["mti"] = mti
                p["mtc"] = mtc
                p["nlc"] = nlc
        if materials is not None:
            p["materials"] = materials
        self._append("DTFR", p)
        return self

    def powr(
        self,
        *,
        ngendf: int | None = None,
        nout: int | None = None,
        lib: int = 1,
        iprint: str | None = None,
        iclaps: int = 0,
        # lib=1 and lib=2: materials list
        materials: list[dict] | None = None,
        # lib=3 (CPM) specific
        nlib: int = 1,
        idat: int = 0,
        iopt: int = 0,
        mode: int = 0,
        if5: int = 0,
        if4: int = 0,
        mat_list: list[str] | None = None,
        nuclides: list[dict] | None = None,
        burnup_data: list | None = None,
    ) -> InputDeck:
        """Add a POWR step (EPRI-CELL / EPRI-CPM library from GENDF)."""
        ngendf = self._resolve_input(ngendf, '_gendf_tape', 'GENDF')
        nout = nout if nout is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "ngendf": ngendf,
            "nout": nout,
            "lib": lib,
            "iclaps": iclaps,
        }
        if iprint is not None:
            p["iprint"] = iprint
        if lib in (1, 2):
            if materials is not None:
                p["materials"] = materials
        elif lib == 3:
            p["nlib"] = nlib
            p["idat"] = idat
            p["iopt"] = iopt
            p["mode"] = mode
            p["if5"] = if5
            p["if4"] = if4
            if mat_list is not None:
                p["mat_list"] = mat_list
            if nuclides is not None:
                p["nuclides"] = nuclides
            if burnup_data is not None:
                p["burnup_data"] = burnup_data
        self._append("POWR", p)
        return self

    def plotr(
        self,
        *,
        nplt: int | None = None,
        nplt0: int = 0,
        plot_def: PlotrJob | None = None,
    ) -> InputDeck:
        """Add a PLOTR step (plot commands from ENDF/PENDF/GENDF tape data).

        Auto-allocates a positive output tape for the plot-command file.
        When *plot_def* is provided the generator emits inline plot
        definition cards after Card 0.
        """
        if nplt is None:
            nplt = abs(self._alloc_tape())
        p: dict[str, Any] = {"nplt": nplt, "nplt0": nplt0}
        if plot_def is not None:
            p["plot_def"] = plot_def
        self._append("PLOTR", p)
        self._plotr_tape = nplt
        return self

    def viewr(
        self,
        *,
        infile: int | None = None,
        nps: int,
        plot_def: ViewrPlot | None = None,
    ) -> InputDeck:
        """Add a VIEWR step.

        When *plot_def* is provided the generator emits inline plot
        commands and forces ``infile=5`` (stdin).  When *infile* is not
        specified and a preceding PLOTR step exists, the PLOTR output
        tape is used automatically.  Otherwise defaults to 5 (stdin).
        """
        if infile is None:
            if plot_def is not None:
                infile = 5
            elif self._plotr_tape is not None:
                infile = self._plotr_tape
            else:
                infile = 5
        p: dict[str, Any] = {"infile": infile, "nps": nps}
        if plot_def is not None:
            p["plot_def"] = plot_def
        self._append("VIEWR", p)
        return self

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _append(self, name: str, params: dict[str, Any]) -> None:
        self._lines.extend(generate_module(name, params))
        self._modules.append(name)

    def render(self) -> str:
        """Return the full NJOY input deck as a string."""
        if not self._lines:
            return ""
        return "\n".join(self._lines + ["stop"])

    def save(self, path: str | Path) -> None:
        """Write the input deck to *path*."""
        Path(path).write_text(self.render(), encoding="utf-8")

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        chain = " -> ".join(self._modules) if self._modules else "empty"
        return f"InputDeck({len(self._modules)} modules: {chain})"
