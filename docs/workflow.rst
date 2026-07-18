Inference workflow
==================

The public model is assembled from three objects:

1. :class:`gapmoe.Histogram` opens one prepared event directory.
2. :class:`gapmoe.Isochrone` describes the source bands and optional selection.
3. :class:`gapmoe.Model` combines them with a :class:`gapmoe.ParamType`.

.. code-block:: python

   import gapmoe

   backend = gapmoe.Histogram.open("runs/event-001")
   source = gapmoe.Isochrone(
       reference_band="Imag",
       color_bands=("Vmag", "Imag"),
       magnitude_range=(15.0, 21.0),
       color_range=(0.5, 3.0),
   )
   model = gapmoe.Model(
       gapmoe.ParamType(parallax=True, distance="sample"),
       l=backend.pre_run.l_deg,
       b=backend.pre_run.b_deg,
       source=source,
       extinction={"Imag": 0.0, "Vmag": 0.0},
       backend=backend,
   )

   print(model.names)

``model.names`` is the exact order expected by ``model.log_density``. For the
parallax, sampled-distance configuration above it is
``(t0, tE, u0, rho, piEN, piEE, DS)``.

``context`` contains values not sampled in the light-curve parameter vector.
For example, parameterizations that use finite-source information need
``thS``; geocentric transformations additionally use the Earth velocity from
:func:`gapmoe.calc_vEarth`.

.. code-block:: python

   logp = model.log_density(theta, context={"thS": theta_star_mas})

Use ``model.to_physical(theta, context=...)`` to inspect the corresponding
``(ML, DL, DS, mu_N, mu_E)`` values. ``model.log_density_batch`` is suitable
for JAX-vectorized evaluation.

The physical model is available as ``model.physical`` for diagnostic work such
as evaluating the five-dimensional histogram density directly. Inference code
should use ``Model`` so the parameterization Jacobian is included.
