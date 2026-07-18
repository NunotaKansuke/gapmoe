Getting started
===============

Installation
------------

Install the package and its bundled ``genulens`` dependency:

.. code-block:: bash

   pip install gapmoe

For local development and documentation builds:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Units and density variables
---------------------------

The histogram density uses ``ML`` in solar masses, ``DL`` and ``DS`` in kpc,
and ``mu_N``/``mu_E`` in mas per year. The public :class:`gapmoe.Model` accepts
light-curve coordinates selected by :class:`gapmoe.ParamType`; it converts them
to these physical variables internally.

Preparing an event
------------------

Only sky coordinates are normally required. The default preprocessor runs
until its statistical error target is met and records all generated artifacts
in a per-event directory.

.. code-block:: python

   from gapmoe.pre_runner import PreRunner

   runner = PreRunner(output_dir="runs")
   pre_run = runner.run(
       ra_deg=270.0,
       dec_deg=-30.0,
       run_name="event-001",
   )

``PreRunner`` also accepts Galactic longitude and latitude through ``l=`` and
``b=``. Supplying mutually inconsistent coordinate aliases raises an error.

For a source checkout of genulens, select the CLI integration explicitly:

.. code-block:: python

   runner = PreRunner(genulens_root="../genulens", backend="cli")
   runner.check_environment()

Advanced preprocessing arguments such as distance ranges, bin widths, and
simulation counts remain available, but should generally be left at their
scientific defaults.
