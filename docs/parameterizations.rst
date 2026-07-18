Parameterizations
=================

``ParamType`` is the stable selector for gapmoe's supported microlensing
coordinate systems. Pass it directly to :class:`gapmoe.Model`.

.. code-block:: python

   import gapmoe

   sampled = gapmoe.ParamType(parallax=True, distance="sample")
   marginalized = gapmoe.ParamType(parallax=True, distance="marginalize")
   no_parallax = gapmoe.ParamType(parallax=False)

The selected object exposes ``names``. Always use that property rather than
hard-coding a sampler vector order.

Distance handling
-----------------

``distance="sample"`` makes ``DS`` an explicit light-curve sampling parameter.
``distance="marginalize"`` integrates it with the histogram distance grid.
Without parallax, the default parameterization marginalizes the hidden physical
distances and proper-motion direction using the histogram backend's native
projections.

The histogram implementation does not support dynamic source-photometry or
additional physical priors while these hidden variables are marginalized. In
that case, sample the relevant distances explicitly.

Orbital models
--------------

Binary circular and Keplerian orbital mappings are selected with
``orbital_motion="circular"`` and ``orbital_motion="kepler"``. They append
derived orbital quantities to the physical mapping while keeping the Galactic
prior in its five-dimensional physical space.
