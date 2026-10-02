# Public input sources

These sources and accessions are transcribed from the recorded research receipts. Exact local source paths, retrieval details and checksums are manifest records.

| Input | Public source / accession |
|---|---|
| BEELINE panels | [BEELINE](https://github.com/murali-group/beeline), [GENELink](https://github.com/zpliulab/GENELink); hHEP GSE81252, hESC GSE75748, mESC GSE98664, mDC GSE48968, mHSC GSE81682. |
| SERGIO | [SERGIO](https://github.com/PayamDiba/SERGIO). Frozen sparse/dense complete truths: 300 genes, nine types, 300 cells per type, 300 / 1,500 edges; generator parameters remain in the source metadata. |
| K562 | Replogle et al., Cell; Figshare record 21632564. [Raw counts](https://ndownloader.figshare.com/files/35775507), [normalized bulk](https://ndownloader.figshare.com/files/35773217), released Anderson–Darling/BH table. |
| RPE1 | Figshare record 20029387. [Raw counts](https://ndownloader.figshare.com/files/35775606), [normalized bulk](https://ndownloader.figshare.com/files/35775512). |
| Primary T cells | GEO GSE190604; [Zenodo record 5784651](https://zenodo.org/records/5784651), CRISPRa-Perturb-seq archive, GEO feature table, donor assignments and guide calls. |
| Promoter binding | [ENCODE](https://www.encodeproject.org), frozen K562 TF experiment metadata and selected accession/MD5 records; GENCODE basic annotation release 44. |
| TRRUST expression | GEO GSE162632, Randolph metadata Figshare 24311335, GENCODE 41 annotation and the recorded Weinstock reference-gene table. Human TRRUST release 2; its original download URL was not recovered. |
| Curated prior | [OmniPath](https://omnipathdb.org): human DoRothEA levels A/B. Use the frozen source-restricted prior and its publication/source audit, not a new unrestricted query. |
| Human TF catalogue | [Human TF database](https://humantfs.ccbr.utoronto.ca/download.php), recorded database extract 1.01. |
| scGPT input | [Checkpoint](https://huggingface.co/wanglab/scGPT-human), [upstream code](https://github.com/bowang-lab/scGPT). Token embeddings, complete union, missing indicators and evaluation-universe mappings are frozen in the bundle. |
| Geneformer input for scRegNet | GF-20L-95M-i4096, identified by the recorded embedding/checkpoint hashes. Original download location remains unresolved. |

The exact raw-to-processed BEELINE recipe and original sampled-split command were not traced. The original ENCODE metadata query, curated-prior writer and TRRUST download receipt also remain unresolved. The release preserves their processed evidence; it does not claim those missing recipes can be reconstructed from upstream defaults.
