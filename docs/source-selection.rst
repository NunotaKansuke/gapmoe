Source selection
================

gapmoe conditions source-distance density on the observed source population.
``PreRunner`` enables the genulens source-selection calculation by default, so
``rho.dat`` contains the ``rhoD_S`` columns used by the histogram backend.

An :class:`gapmoe.Isochrone` can apply a fixed source selection:

.. code-block:: python

   source = gapmoe.Isochrone(
       reference_band="Imag",
       color_bands=("Vmag", "Imag"),
       magnitude_range=(15.0, 21.0),
       color_range=(0.5, 3.0),
   )

Alternatively, evaluate the event prior conditional on current measured
photometry by passing named magnitudes to ``model.log_density``:

.. code-block:: python

   model.log_density(
       theta,
       context={"thS": theta_star_mas},
       magnitudes={"Imag": i_s, "Vmag": v_s},
   )

This is a conditional event density. Use ``model.log_joint_density`` when the
source-photometry factor itself belongs in the target density.

``model.source_radius`` and ``model.log_theta_star_density`` expose the source
population summaries associated with the same CMD model.
