Reading a file
==============

One door
--------

:func:`kika.read` opens any nuclear data file kika understands and gives back the
same kind of object every time:

.. code-block:: python

   import kika

   ev = kika.read("Fe56.endf")     # or an ACE file
   ev.reactions[102]               # capture, by MT
   ev.reactions["capture"]         # the same reaction, by GNDS label
   E, xs = ev.cross_section(102)   # two numpy arrays

The format is detected from the file's **content**, not its extension — ``.dat``,
``.txt``, no extension at all and ACE files named after their ZAID are all
routine in this field. Pass ``format="endf"`` to override the detection for a
file with a malformed header.

What comes back
---------------

A ``ReactionSuite``: kika's canonical model, whose structure and vocabulary
follow the **GNDS-2.1** specification. Node names are the spec's, verbatim, so
what you read in the standard is what you type::

   ev.reactions[102].crossSection.evaluated
   ev.resonances.resolved[0].formalism
   ev.covarianceSuite.covarianceSections

Method names are Python's — ``ev.cross_section(102)``, ``ev.summary()`` — so a
verb is always ``snake_case`` and a GNDS node never is.

Start with ``summary()``, which says what the file actually contained:

.. code-block:: text

   >>> print(ev.summary())
   n + Fe56   [JEFF-4.0]  GNDS 2.1
     styles       ['eval']
     reactions    37  MT[1, 2, 4, 16, ...]
     PoPs         1 particles
     resonances   1 resolved, 1 unresolved   [1e-05 eV - 850 keV]
     covariances  21 sections
     decode       2 unsupported

Check what was dropped
----------------------

That last line matters. kika does not yet model every ENDF section — MF1, 2,
3, 4, 5, 6 and the covariance files 31-35 are decoded into the model; MF7 and
MF12-15 are not read at all — so a decode is often **partial**, and
``ev.report`` is the only thing that says so:

.. code-block:: python

   ev.report.isClean          # nothing lost, approximated or refused
   ev.report.losses           # data in the file that is not in the result
   ev.report.approximations   # data that is in the result but changed on the way
   ev.report.unsupported      # constructs kika recognised and cannot handle

``approximations`` is the one to read first. A loss is visible — the field is
missing. An approximation looks like data.

The other road
--------------

:func:`kika.read` is the default, not the only way in. :func:`kika.read_endf`
and :func:`kika.read_ace` give you the file in **its own terms** — MF/MT
sections for ENDF, blocks for ACE — and they are fully supported, not
deprecated:

.. code-block:: python

   endf = kika.read_endf("Fe56.endf")
   endf.mf[33].mt[102]        # the covariance section as ENDF structures it

Reach for them when you need a section kika does not model yet, when you are
working as an evaluator in ENDF's own vocabulary, or when you are writing a
tape back out. They are also the **faster** road: decoding into the model costs
roughly 2.5x a bare parse, so production pipelines that read thousands of tapes
should keep using ``read_endf``.

What of GNDS is supported
-------------------------

GNDS files are read and written. What that does **not** mean is that kika
implements GNDS 2.1 — it reads the parts the ENDF/B-VIII.1 neutron
evaluations use, and the difference is large enough to be worth stating in
full rather than glossing:

.. code-block:: python

   >>> import kika.gnds
   >>> print(kika.gnds.capabilities().summary())
   300 of GNDS's nodes: 134 full, 7 partial, 159 unsupported (17 lost without
   a report line); and 12 nodes kika names that gnds.xsd does not declare

The left-hand column is every element ``gnds.xsd`` and ``covariances.xsd``
declare, so a node kika does not touch is listed as unsupported rather than
being absent. Each row says why, citing a section of the specification or a
line of the source:

.. code-block:: python

   >>> print(kika.gnds.capabilities(group="thermalScattering").text())
   >>> print(kika.gnds.capabilities(coverage="partial").text())

The covariance chapter, §25, is the one kika covers completely. The thermal
scattering law and the double-differential cross sections are not read at all.
``capabilities().silent`` is the short list of nodes that are dropped with
nothing said — everything else that is lost produces a line in ``ev.report``.

``capabilities()`` says what the *library* can lose, without opening a file.
``ev.report`` says what *your* file lost. Neither answers for the other.
