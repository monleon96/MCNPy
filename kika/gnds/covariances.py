"""GNDS §25 ``covarianceSuite`` → the model.

The half of a GNDS evaluation that matters most here, and the one whose
vocabulary is fully known: the tables below were measured across all 270
covariance files of ENDF/B-VIII.1-GNDS, so "the reader handles what occurs" is
a statement about the data rather than about the specification.

**Array storage is transcribed from FUDGE, not inferred.** ``array`` has three
storage modes and getting any of them subtly wrong yields a plausible matrix
rather than an error, so each is written against
``/soft_snc/FUDGE/6.10.0/fudge/xData/xDataArray.py`` — the code that *wrote*
these files — with the class and method named in the comment. The one that
cannot be guessed from the data is ``flattened``: its ``starts`` are flat
indices into the **full** ``n*m`` array even when ``symmetry="lower"`` is also
set, which is why the mirroring happens after the reshape and not before.

**A ``covarianceSuite`` read alone is a supported state.** Four of the committed
fixtures are covariance files without their ``reactionSuite`` sibling, and a
user handed one file is in the same position. Every ``rowData`` href that cannot
be followed becomes a report entry naming the link; nothing raises, and the
matrices are read either way. What is lost is only the ability to say *which
cross section in which file* a section is about — and the ``ENDF_MFMT``
attribute beside the href still answers that in ENDF's terms.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from kika.gnds.nodes import reads, readersOf
from kika.nuclear_data.model import (AverageParameterCovariance,
                                     ConversionReport, CovarianceMatrix,
                                     CovarianceSection, CovarianceSuite,
                                     DataLink, Mixed, ParameterCovariance,
                                     ParameterCovarianceMatrix, ParameterLink,
                                     ShortRangeSelfScalingVariance, Slice,
                                     Slices, Sum, Summand)

from .primitives import readAxes, readValues
from .styles import readStyles
from .version import checkFormat
from .xpath import Document, Resolver, readExternalFiles

__all__ = ["readArray", "readCovarianceSuite"]

#: §25. The forms a ``covarianceSection`` may hold, with the reader for each.
#: Filled at the bottom of the module, once the readers exist.
COVARIANCE_FORMS: Dict[str, Callable] = {}

#: §25.3. The two kinds of node ``<parameterCovariances>`` holds. Same
#: arrangement and same reason: derived from ``nodes.py`` rather than written
#: out a second time here.
PARAMETER_COVARIANCE_FORMS: Dict[str, Callable] = {}


# ---------------------------------------------------------------------------
# array
# ---------------------------------------------------------------------------

def readArray(element: ET.Element) -> np.ndarray:
    """§25's ``array`` → a dense 2-D numpy array.

    Three storage modes, each transcribed from the FUDGE class that writes it:

    ``compression`` absent
        ``xDataArray.Full``. Either ``n*m`` values row-major, or — with
        ``symmetry="lower"`` — the ``n(n+1)/2`` lower-triangle entries, which
        are placed with ``tril_indices`` and mirrored into the upper triangle.

    ``compression="diagonal"``
        ``xDataArray.Diagonal``. The main diagonal only, ``n`` values. FUDGE
        supports several diagonals through a ``startingIndices`` child; no file
        in the distribution has one, so meeting one raises rather than being
        read as the main diagonal.

    ``compression="flattened"``
        ``xDataArray.Flattened``. Sparse runs: ``starts`` gives each run's flat
        index and ``lengths`` its length. **The indices are into the full
        ``n*m`` array**, so the reshape happens first and the ``symmetry``
        mirroring second — ``numpy.tril(a) + numpy.tril(a, -1).T``, exactly as
        ``Flattened.constructArray`` does it.
    """
    shape = tuple(int(part) for part in element.attrib["shape"].split(","))
    if len(shape) != 2:
        raise ValueError(
            f"kika reads two-dimensional covariance arrays; this one declares "
            f"shape={element.attrib['shape']!r}"
        )
    symmetry = element.attrib.get("symmetry")
    compression = element.attrib.get("compression")

    if compression is None:
        return _readFull(element, shape, symmetry)
    if compression == "diagonal":
        return _readDiagonal(element, shape, symmetry)
    if compression == "flattened":
        return _readFlattened(element, shape, symmetry)
    raise ValueError(
        f"array compression={compression!r} is not one of GNDS's "
        f"'diagonal' or 'flattened'"
    )


def _labelledValues(element: ET.Element) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """The ``values`` children, split into the labelled ones and the data."""
    labelled: Dict[str, np.ndarray] = {}
    data: Optional[np.ndarray] = None
    for child in element.findall("values"):
        label = child.attrib.get("label")
        if label is None:
            data = readValues(child)
        else:
            labelled[label] = readValues(child)
    return labelled, data


def _mirror(matrix: np.ndarray, symmetry: Optional[str]) -> np.ndarray:
    """Reflect a stored triangle into the other one. ``Flattened.constructArray``."""
    if symmetry == "lower":
        return np.tril(matrix) + np.tril(matrix, -1).T
    if symmetry == "upper":
        return np.triu(matrix) + np.triu(matrix, -1).T
    return matrix


def _readFull(element: ET.Element, shape, symmetry) -> np.ndarray:
    _, data = _labelledValues(element)
    if data is None:
        raise ValueError("an array with no compression has no values to read")

    if symmetry is None:
        if data.size != shape[0] * shape[1]:
            raise ValueError(
                f"an uncompressed {shape[0]}x{shape[1]} array needs "
                f"{shape[0] * shape[1]} values, not {data.size}"
            )
        return data.reshape(shape)

    # `Full.constructArray`: the triangle is written out, then reflected.
    rows = shape[0]
    expected = rows * (rows + 1) // 2
    if data.size != expected:
        raise ValueError(
            f"a {symmetry}-triangular {rows}x{rows} array needs {expected} "
            f"values, not {data.size}"
        )
    matrix = np.zeros(shape, dtype=float)
    indices = (np.tril_indices(rows) if symmetry == "lower"
               else np.triu_indices(rows))
    matrix[indices] = data
    return _mirror(matrix, symmetry)


def _readDiagonal(element: ET.Element, shape, symmetry) -> np.ndarray:
    if element.find("values[@label='startingIndices']") is not None:
        raise ValueError(
            "this diagonal array declares startingIndices, so it stores more "
            "than the main diagonal. No file in ENDF/B-VIII.1-GNDS does, so "
            "kika does not read it — reading it as the main diagonal would put "
            "off-diagonal variance on the diagonal without saying so"
        )
    _, data = _labelledValues(element)
    if data is None:
        raise ValueError("a diagonal array has no values to read")

    length = min(shape)
    if data.size != length:
        raise ValueError(
            f"the main diagonal of a {shape[0]}x{shape[1]} array is {length} "
            f"values, not {data.size}"
        )
    matrix = np.zeros(shape, dtype=float)
    np.fill_diagonal(matrix, data)
    return matrix


def _readFlattened(element: ET.Element, shape, symmetry) -> np.ndarray:
    labelled, data = _labelledValues(element)
    try:
        starts, lengths = labelled["starts"], labelled["lengths"]
    except KeyError as exc:
        raise ValueError(
            f"a flattened array needs values labelled 'starts' and 'lengths'; "
            f"this one has {sorted(labelled)}"
        ) from exc
    if data is None:
        raise ValueError("a flattened array has starts and lengths but no data")
    if starts.size != lengths.size:
        raise ValueError(
            f"{starts.size} starts against {lengths.size} lengths"
        )
    if int(lengths.sum()) != data.size:
        raise ValueError(
            f"the run lengths account for {int(lengths.sum())} values, "
            f"the array stores {data.size}"
        )

    # `Flattened.constructArray`: a flat buffer of n*m, filled run by run. The
    # starts index this buffer and NOT the stored triangle, even when symmetry
    # is 'lower' -- which is why the mirroring is applied after the reshape.
    flat = np.zeros(shape[0] * shape[1], dtype=float)
    cursor = 0
    for start, length in zip(starts.tolist(), lengths.tolist()):
        flat[start:start + length] = data[cursor:cursor + length]
        cursor += length
    return _mirror(flat.reshape(shape), symmetry)


# ---------------------------------------------------------------------------
# gridded2d and the forms built on it
# ---------------------------------------------------------------------------

def _readGridded2d(element: ET.Element, resolve: Optional[Callable]):
    """``gridded2d`` → ``(matrix, rowGrid, columnGrid, unit)``.

    §5.1.3's grids carry the *boundaries* of the bins, so a grid has one more
    value than the matrix has rows — which is exactly what
    :class:`CovarianceMatrix` checks.
    """
    axes = readAxes(element, resolve)
    array = element.find("array")
    if array is None:
        raise ValueError("a gridded2d with no array")
    matrix = readArray(array)

    def gridValues(index: int) -> Optional[np.ndarray]:
        if axes is None:
            return None
        try:
            return getattr(axes.byIndex(index), "values", None)
        except KeyError:
            return None

    # §25: index 2 is the row axis and index 1 the column axis, the same
    # highest-index-first convention §5.1.1 gives every other function.
    return matrix, gridValues(2), gridValues(1), axes


@reads("covarianceForm", "covarianceMatrix")
def _readCovarianceMatrix(element: ET.Element, resolve, report) -> CovarianceMatrix:
    gridded = element.find("gridded2d")
    if gridded is None:
        raise ValueError(
            f"a covarianceMatrix holding no gridded2d: {sorted(c.tag for c in element)}"
        )
    matrix, rowGrid, columnGrid, _ = _readGridded2d(gridded, resolve)
    return CovarianceMatrix(
        matrix=matrix,
        rowGrid=rowGrid,
        columnGrid=columnGrid,
        isRelative=element.attrib.get("type") == "relative",
        label=element.attrib.get("label"),
        productFrame=element.attrib.get("productFrame"),
    )


@reads("covarianceForm", "shortRangeSelfScalingVariance")
def _readShortRange(element: ET.Element, resolve, report) -> ShortRangeSelfScalingVariance:
    return ShortRangeSelfScalingVariance(
        matrix=_readCovarianceMatrix(element, resolve, report),
        dependenceOnProcessedGroupWidth=element.attrib.get(
            "dependenceOnProcessedGroupWidth"
        ),
        label=element.attrib.get("label"),
    )


@reads("covarianceForm", "mixed")
def _readMixed(element: ET.Element, resolve, report) -> Mixed:
    """§25. Components that together make one covariance.

    A ``mixed`` is *not* simply "add these matrices": its components sit on
    different grids, and one of them may be a
    :class:`ShortRangeSelfScalingVariance`, whose magnitude depends on the
    processing group width. They are kept as the list the file gives, and
    combining them is the caller's decision.
    """
    components = []
    for child in element:
        reader = COVARIANCE_FORMS.get(child.tag)
        if reader is None:
            report.unsupportedNode(
                f"mixed[{element.attrib.get('label')!r}] holds a <{child.tag}>, "
                f"which kika does not read; that component is absent from the "
                f"covariance and the remaining ones do not add up to it"
            )
            continue
        components.append(reader(child, resolve, report))
    return Mixed(components=components, label=element.attrib.get("label"))


@reads("covarianceForm", "sum")
def _readSum(element: ET.Element, resolve, report) -> Sum:
    return Sum(
        summands=[
            Summand(
                href=child.attrib.get("href", ""),
                coefficient=float(child.attrib.get("coefficient", 1.0)),
                ENDF_MFMT=child.attrib.get("ENDF_MFMT"),
            )
            for child in element.findall("summand")
        ],
        label=element.attrib.get("label"),
        domainMin=_optionalFloat(element, "domainMin"),
        domainMax=_optionalFloat(element, "domainMax"),
        domainUnit=element.attrib.get("domainUnit", ""),
    )


def _optionalFloat(element: ET.Element, name: str) -> Optional[float]:
    value = element.attrib.get(name)
    return None if value is None else float(value)


# Derived from `kika/gnds/nodes.py` rather than listed a second time here --
# see the note on `primitives.FUNCTION_1D`. Still an `.update` on the dict
# declared at the top of the module, because `_readMixed` closes over that
# object to read its own components.
COVARIANCE_FORMS.update(readersOf("covarianceForm"))


# ---------------------------------------------------------------------------
# rowData / columnData
# ---------------------------------------------------------------------------

def _readDataLink(element: ET.Element) -> DataLink:
    """§25.2.3-4. What a matrix is *about*, plus any slices restricting it."""
    entries = []
    for container in element.findall("slices"):
        for child in container.findall("slice"):
            entries.append(Slice(
                dimension=int(child.attrib["dimension"]),
                domainValue=_optionalFloat(child, "domainValue"),
                domainMin=_optionalFloat(child, "domainMin"),
                domainMax=_optionalFloat(child, "domainMax"),
                domainUnit=child.attrib.get("domainUnit", ""),
            ))
    return DataLink(
        href=element.attrib.get("href", ""),
        ENDF_MFMT=element.attrib.get("ENDF_MFMT"),
        dimension=(int(element.attrib["dimension"])
                   if "dimension" in element.attrib else None),
        slices=Slices(entries),
    )


def _noteUnfollowableLink(link: Optional[DataLink], resolver: Optional[Resolver],
                          owner: str, report: ConversionReport) -> None:
    """Report a ``rowData``/``columnData`` href that reaches nothing.

    Reported once per link and never raised: a covariance file read without its
    ``reactionSuite`` sibling has one of these per section, which is a normal
    state and not a defect in the file.
    """
    if link is None or not link.href or resolver is None:
        return
    resolution = resolver.resolve(link.href)
    if not resolution:
        report.lost(
            f"{owner}: {resolution.problem}. The covariance itself is read; "
            f"what is missing is the link to the quantity it is about"
            + (f" (ENDF {link.ENDF_MFMT})" if link.ENDF_MFMT else "")
        )


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _readSection(element: ET.Element, resolve, resolver, report) -> CovarianceSection:
    label = element.attrib.get("label", "")
    rowData = element.find("rowData")
    columnData = element.find("columnData")

    form = None
    for child in element:
        reader = COVARIANCE_FORMS.get(child.tag)
        if reader is not None:
            form = reader(child, resolve, report)
            break
    else:
        known = [c.tag for c in element if c.tag not in ("rowData", "columnData")]
        if known:
            report.unsupportedNode(
                f"covarianceSection {label!r} holds {known}, none of which is a "
                f"covariance form kika reads; the section has no matrix"
            )

    section = CovarianceSection(
        label=label,
        rowData=None if rowData is None else _readDataLink(rowData),
        columnData=None if columnData is None else _readDataLink(columnData),
        form=form,
        crossTerm=element.attrib.get("crossTerm") == "true",
    )
    _noteUnfollowableLink(section.rowData, resolver,
                          f"covarianceSection {label!r} rowData", report)
    _noteUnfollowableLink(section.columnData, resolver,
                          f"covarianceSection {label!r} columnData", report)
    return section


@reads("parameterCovarianceForm", "parameterCovariance")
def _readParameterCovariance(element: ET.Element, resolve, resolver, report):
    label = element.attrib.get("label", "")
    rowData = element.find("rowData")
    columnData = element.find("columnData")

    matrix = element.find("parameterCovarianceMatrix")
    form = None
    if matrix is not None:
        form = _readParameterCovarianceMatrix(matrix, report)
    else:
        report.unsupportedNode(
            f"parameterCovariance {label!r} holds no parameterCovarianceMatrix"
        )

    covariance = ParameterCovariance(
        label=label,
        rowData=None if rowData is None else _readDataLink(rowData),
        columnData=None if columnData is None else _readDataLink(columnData),
        form=form,
    )
    _noteUnfollowableLink(covariance.rowData, resolver,
                          f"parameterCovariance {label!r} rowData", report)
    return covariance


def _readParameterCovarianceMatrix(element: ET.Element,
                                   report) -> ParameterCovarianceMatrix:
    """§25.3.2. The matrix plus the parameter runs that index its rows.

    **``matrixStartIndex`` is already zero-based, and the arithmetic says so.**
    Si-32's matrix is 19x19 and carries two links: ``scatteringRadius`` with no
    attributes at all (so one parameter, starting at row 0) and
    ``resonanceParameters`` with ``nParameters="18" matrixStartIndex="1"``.
    1 + 18 = 19 only if row 1 is the *second* row. Subtracting one to "convert
    from one-based" — which is what this function did first — puts both links at
    row 0, and since the model only checks that the counts add up to the matrix
    order, nothing would have complained.
    """
    array = element.find("array")
    if array is None:
        raise ValueError("a parameterCovarianceMatrix with no array")

    links: List[ParameterLink] = []
    for container in element.findall("parameters"):
        for child in container.findall("parameterLink"):
            links.append(ParameterLink(
                label=child.attrib.get("label", ""),
                href=child.attrib.get("href", ""),
                nParameters=int(child.attrib.get("nParameters", 1)),
                matrixStartIndex=int(child.attrib.get("matrixStartIndex", 0)),
            ))

    matrix = readArray(array)
    declared = sum(link.nParameters for link in links)
    if links and declared != matrix.shape[0]:
        # The model refuses this combination outright, and it is right to: a
        # link set that does not account for every row cannot say what row 47
        # is. Reported and dropped rather than raised, so one malformed section
        # does not cost the reader the other ten.
        report.lost(
            f"parameterCovarianceMatrix {element.attrib.get('label')!r}: its "
            f"parameterLinks account for {declared} rows and the matrix has "
            f"{matrix.shape[0]}, so the rows cannot be named; the links are "
            f"dropped and the matrix is kept"
        )
        links = []

    return ParameterCovarianceMatrix(
        matrix=matrix,
        parameters=links,
        isRelative=element.attrib.get("type") == "relative",
        label=element.attrib.get("label"),
    )


@reads("parameterCovarianceForm", "averageParameterCovariance")
def _readAverageParameterCovariance(element: ET.Element, resolve, resolver, report):
    label = element.attrib.get("label", "")
    rowData = element.find("rowData")
    columnData = element.find("columnData")

    form = None
    matrix = element.find("covarianceMatrix")
    if matrix is not None:
        form = _readCovarianceMatrix(matrix, resolve, report)
    else:
        report.unsupportedNode(
            f"averageParameterCovariance {label!r} holds no covarianceMatrix"
        )

    covariance = AverageParameterCovariance(
        label=label,
        rowData=None if rowData is None else _readDataLink(rowData),
        columnData=None if columnData is None else _readDataLink(columnData),
        form=form,
        crossTerm=element.attrib.get("crossTerm") == "true",
    )
    _noteUnfollowableLink(covariance.rowData, resolver,
                          f"averageParameterCovariance {label!r} rowData", report)
    return covariance


PARAMETER_COVARIANCE_FORMS.update(readersOf("parameterCovarianceForm"))


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------

def readCovarianceSuite(
    document: Document,
    documents: Optional[Dict[str, Document]] = None,
    report: Optional[ConversionReport] = None,
) -> Tuple[CovarianceSuite, ConversionReport]:
    """§25.1.1 ``covarianceSuite`` → the model, plus what the read could not do.

    Parameters
    ----------
    document
        The parsed covariance file.
    documents
        ``externalFile`` label → parsed :class:`Document`, for following
        ``$reactions#`` links. Omitting it is supported and normal; every link
        that then cannot be followed becomes an entry in the report.
    """
    report = report if report is not None else ConversionReport()
    root = document.root
    if root.tag != "covarianceSuite":
        raise ValueError(
            f"{document.name} is a <{root.tag}>, not a <covarianceSuite>"
        )
    checkFormat(root.attrib.get("format"), document.name)

    resolver = Resolver(document, documents)
    resolve = lambda href, node: resolver.resolve(href, context=node).element  # noqa: E731

    suite = CovarianceSuite(
        evaluation=root.attrib.get("evaluation"),
        projectile=root.attrib.get("projectile"),
        target=root.attrib.get("target"),
        interaction=root.attrib.get("interaction", "nuclear"),
        externalFiles=readExternalFiles(root),
        covarianceSections=[],
        parameterCovariances=[],
    )

    styles = root.find("styles")
    if styles is not None:
        # §25.1.1 makes `styles` mandatory on this root too, and every form in
        # the file is labelled with one of them. Read for the same reason as on
        # the reactionSuite, and because a file written without it is invalid.
        suite.styles = readStyles(styles, f"{document.name}/styles", report)

    sections = root.find("covarianceSections")
    for child in (sections if sections is not None else []):
        if child.tag != "covarianceSection":
            report.unsupportedNode(
                f"<covarianceSections> holds a <{child.tag}>, which kika does "
                f"not read"
            )
            continue
        suite.covarianceSections.append(
            _readSection(child, resolve, resolver, report)
        )

    parameters = root.find("parameterCovariances")
    for child in (parameters if parameters is not None else []):
        reader = PARAMETER_COVARIANCE_FORMS.get(child.tag)
        if reader is None:
            report.unsupportedNode(
                f"<parameterCovariances> holds a <{child.tag}>, which kika does "
                f"not read"
            )
            continue
        suite.parameterCovariances.append(
            reader(child, resolve, resolver, report)
        )

    # Both ways to the same object: the tuple is unchanged, and the
    # attribute is the one that survives `suite, _ = ...`. §11.4.
    suite.report = report
    return suite, report
