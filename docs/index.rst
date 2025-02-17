.. _assessment:

****************
Model assessment
****************

Thermodynamics-based model assessment consists in finding parts of the network where the
model structure and the measured or assumed metabolite concentrations are not
compatible, suggesting model inaccuracies or novel mechanisms. Such cases can have
different explanations:

* COBRA models have been constructed with Flux Balance Analysis (FBA) as the main
  application and frequently contain thermodynamic inaccuracies. Some reactions may be
  annotated as irreversible in the model, while they are not in practice.
* The direct interpretation of COBRA models is that individual reactions are independent
  events, i.e. (1) substrates diffuse to the enzyme, (2) substrates are converted to
  products (3) products equilibrate with the compartment. However the actual mechanism
  could be different. For example, substrate channeling could occur. In that case, an
  intermediate is passed directly from one enzyme to the other, without equilibrating
  with the metabolite pool in the compartment. While this mechanism is irrelevant for
  FBA, it has significant influence on thermodynamics.
* Despite continuous improvements, we still don't know the exact stoichiometry of
  metabolic networks, even for well-studies organisms. Reactions may be missing or have
  incorrect cofactors (with different thermodynamic properties).

Additionally, model assessment can tell us which reactions impose the strongest
constraints on the thermodynamics of the network.

See :footcite:`gollub2020probabilistic` for additional information.

Thermodynamic anomalies
-----------------------

We can use PMO to detect *anomalies* in the network, i.e. variables whose predicted
value is significantly different from its prior. A predicted value is flagged as anomaly
if its `z-score <https://en.wikipedia.org/wiki/Standard_score>`_ is larger than a
certain threshold. A threshold of one means that a variable is flagged only if its
predicted value is at least one standard deviation away from the its initial estimate.

Identifying anomalies
---------------------

The :code:`QuantitativeAssessment` class can be used to detect anomalies in a PMO
solution. While you can set arbitrary objective in PMO, we recommend to use the default
objective (maximize the probability of the thermodynamic variables).

.. code-block:: python
    :linenos:

    # Construct and solve the PMO problem.
    problem = pta.PmoProblem(model, thermodynamic_space)
    problem.solve()

    # Analyze the result to find anomalies in the predicted values.
    assessment = pta.QuantitativeAssessment(problem)
    assessment.summary()

.. code-block::

    Quantitative thermodynamic assessment summary:
    ------------------------------------------------
    conentrations: mM, free energies: kJ/mol

    > The following metabolites have been flagged as anomalies because their predicted
      concentration has an absolute z-score greater than 1.0:
                id       conc  z_log_c
    524     mqn8_c  3.831e+03    4.843
    520     mql8_c  1.434e-05   -4.843
    234    fadh2_c  2.214e-05   -4.627
    544     iasp_c  7.845e-05   -3.995
    ...

    > No anomaly found in non-intracellular concentrations.

    > The following reactions have been flagged as anomalies because their predicted
      free energy or standard free energy has an absolute z-score greater than 1.0:
                id          v     drg0        drg  z_drg     z_drg0     sp_drg
    178         ASPO4  2.020e-08   84.123 -1.000e-03 -9.908 -5.824e+00  1.547e+00
    479       GLYCTO3  1.373e-07   59.109 -1.000e-03 -7.215 -4.758e+00  2.611e-01
    378          FRD2  7.783e-07  -75.359 -2.391e+01  6.513  4.571e+00  4.807e-12
    ...

    > The following reactions are predicted to impose strong thermodynamic constraints
      on the network because their shadow price is greater than 0.1:
            id          v    drg0        drg  z_drg     z_drg0  sp_drg
    178    ASPO4  2.020e-08  84.123 -1.000e-03 -9.908 -5.824e+00   1.547
    127    AIRC3 -2.692e-01 -30.990  1.000e-03  5.580  3.627e+00   1.161
    588     IMPD  1.462e-01  48.407 -1.000e-03 -5.900 -4.249e+00   0.991
    ...

Note that you can adjust the threshold for anomalies in the constructor of the
:code:`QuantitativeAssessment` class.

Interpreting and curating anomalies
-----------------------------------

Once you obtain the predicted anomalies you should investigate what is the reason. Here
is a checklist of the most common reasons for anomalies:

1. Is the anomaly expected? E.g. oxygen concentration is often flagged as anomaly, but it
   is normal to expect low concentrations.
2. If the reaction is irreversible, is this annotation correct? Sometimes COBRA models
   are too restrictive in that sense.
3. Is the metabolite channeled between two reactions? Or is there at least some hint for
   that in the literature or in databases such as `STRING <https://string-db.org/>`_?
4. Is the stoichiometry of the reaction correct? Maybe a reaction can use different
   cofactors. Moreover, some core models may use a single electron acceptor for all
   reactions, even if different ones are used in practice.
5. How reliable is the standard reaction energy for the reaction? Free energies for
   exotic or large compounds are determined with group contribution, which may
   underestimate the uncertainty.
6. Is the structure of the metabolite defined? Complex sugars such as glycogen can have
   different structures in different databases.

Points 3 and 4 are particularly interesting, as they can reveal new biological
mechanisms. See :footcite:`gollub2020probabilistic` for additional information.

The snippet above shows part of the output obtained running the assessment on the
iJO1366 model. It shows anomalies in the concentrations of :code:`mqn8`, :code:`mql8`
and :code:`iasp` as well as in the standard free energy of of the reaction they
participate to (aspartate oxidase, :code:`ASPO4`). In this case the most probable
explanation is that iminoaspartate (:code:`iasp`) is channeled between aspartate oxidase
and quinolinate synthase (:footcite:`marinoni2008characterization`).

We can now curate the model by replacing the two reactions with a lumped reaction
representing the channel. Since aspartate oxidase can use different electron acceptors,
we should do the same with all variants of the reaction.

We recommend investigating and resolving anomalies iteratively: start from the highest
anomaly and either resolve it or accept it. If you modified the model, run the
assessment again and continue with the next highest anomaly.

Performance considerations
--------------------------

Running PMO can take a considerable amount of time on medium-large models. The runtime
is even higher when the model contains several anomalies because the solution is pushed
towards regions of low probability (further away from the objective). In these cases we
recommend to separate the process in two steps:

1. Run the assessment and curate the model for each compartment independently. For
   example you can construct a thermodynamic space that only covers intracellular
   reactions as follows:

   .. code-block:: python
      :linenos:

      # Get candidate thermodynamic constraints for the model, excluding
      # extrcellular and periplasmic reactions.
      constrained_rxn_ids = pta.get_candidate_thermodynamic_constraints(
          model,
          metabolites_namespace="bigg.metabolite",
          exclude_compartments=['p', 'e']
      )

      # Construct a thermodynamic space covering only the selected reactions.
      thermodynamic_space = pta.ThermodynamicSpace.from_cobrapy_model(
          model,
          metabolites_namespace="bigg.metabolite",
          constrained_rxns=constrained_rxn_ids
      )

2. After curating each compartment, you can apply PMO and model assessment on the entire
   network. This should now be much faster.

References
----------

.. footbibliography::