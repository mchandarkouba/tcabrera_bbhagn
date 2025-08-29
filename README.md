# BBHAGN

This code analyzes the association between gravitational wave events and active galactic nuclei flares, following the methods of [Palmese+21](https://ui.adsabs.harvard.edu/abs/2021ApJ...914L..34P/abstract).

GW dataframe columns:
- `skymap_path`
- GWTC mass(es), as required by `bbhmass_type` in `config.yaml`
- `f_cover` (fraction of GW posterior covered by followup)
- 
Flare dataframe columns:
- `ra`
- `dec`
- `Redshift`
- `t_peak_{f}`
- `f_peak_{f}`
- `t_rise_{f}`
- `t_decay_{f}`
- `f_base_{f}`

The filter-specific fields in the flare dataframe should be present for at least `f` = `g` and `r`.
They correspond to the model defined by `utils.flaremorphology.graham23_flare_model` (gaussian rise, exponential decay); where relevant, the onset of the flare is assumed to be $t_0$ = `t_peak_g` - 3 * `t_rise_g`.
