# Licence for data, results and trained policies

The source code in this repository is licensed under the MIT License (see `LICENSE`).

The following are licensed under the Creative Commons Attribution 4.0 International
License (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/:

- `results/`: evaluation outputs;
- `checkpoints/`: trained policies;
- `data/`: evaluation-window definitions and RSU data.

Copyright (c) 2026 Anonymous authors. Please cite the accompanying paper when using
these materials.

## Third-party scenario files

The files in `sumo/` (road network, bus-stop definitions and the corridor demand cut
from the scenario's route file) are derived from the Berlin SUMO Traffic scenario (BeST)
of Schrab et al., "A Large-Scale Traffic Scenario of Berlin for Evaluating Smart
Mobility Applications" (2023), https://doi.org/10.1007/978-3-031-23721-8_24,
https://github.com/mosaic-addons/best-scenario. They are redistributed under the
scenario's own licence, CC BY 4.0, with attribution to its authors. They are not
covered by the MIT License.
