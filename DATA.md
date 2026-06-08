# Data Sources

This repository includes the processed benchmark CSV files used by the default
Beacon runs under `data/raws/`. Large generated outputs are not included.

## Included benchmark files

Beacon expects paired expression and network files:

```text
data/raws/<CELL_TYPE>-ExpressionData.csv
data/raws/<CELL_TYPE>-network.csv
data/raws/TFs+500/<CELL_TYPE>-ExpressionData.csv
data/raws/TFs+500/<CELL_TYPE>-network.csv
```

The tracked benchmark files cover hESC, hHEP, mDC, mESC, mHSC-E, mHSC-GM,
mHSC-L, plus human and mouse aggregate files.

## Upstream sources

These processed files follow the GENELink / BEELINE gene regulatory network
benchmark format.

- GENELink repository: https://github.com/zpliulab/GENELink
- BEELINE repository: https://github.com/murali-group/beeline
- BEELINE input documentation: https://murali-group.github.io/Beeline/reproducing-results.html

The original single-cell datasets used by this benchmark family are available
from GEO:

- hHEP: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81252
- hESC: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748
- mESC: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98664
- mDC: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48968
- mHSC: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81682

## Generated files

The following paths are generated locally and are ignored by git:

- `data/splits/`
- `data/raws/prev/`
- `logs/`
- `results/`
