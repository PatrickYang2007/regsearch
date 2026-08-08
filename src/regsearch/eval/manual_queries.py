"""Query set for the hand-judged evaluation.

Why these are not the citation-derived queries
----------------------------------------------
The automatic eval set uses paper TITLES as queries and the papers they cite as
positives. That is a defensible weak-supervision signal, but it is not the task
the system is actually for, and it is awkward to judge: deciding whether paper A
"should" cite paper B is a different question from whether B answers a search.

These are written as things a regulatory-genomics researcher would actually type
into a search box -- questions and topic phrases, not titles. Judging them is
also far easier, which matters because judging is human time and human time is
the bottleneck on every defensible number in this project.

Grounding
---------
Every query was written against the corpus's actual topic distribution, sampled
with ts_stat over document titles: chromatin (2,750 docs), enhancers (2,104),
CTCF (835), cohesin (215), ATAC-seq (193), ChIP-seq (176), Hi-C (121), TADs
(100), GWAS (162), deep learning (238), single-cell (1,333), cancer (1,609).
A judging pool full of queries with no relevant documents would waste the
judge's time and depress every arm equally, measuring nothing.

This grounding is deliberately NOT the same as guaranteeing a good result: the
queries are phrased independently of how any arm retrieves, and no query was
adjusted after seeing what came back. Tuning queries against retrieved output
would make the evaluation circular.
"""

from __future__ import annotations

# Mechanism questions -- the "how does X work" shape, usually answered by a
# review or a mechanistic study.
MECHANISM = [
    "how does CTCF establish topologically associating domain boundaries",
    "cohesin loop extrusion mechanism and its role in gene regulation",
    "how do enhancers physically contact their target promoters",
    "role of polycomb repressive complexes in developmental gene silencing",
    "how does DNA methylation at promoters repress transcription",
    "nucleosome positioning and its effect on transcription factor binding",
    "mechanism of X chromosome inactivation and lncRNA involvement",
    "how do pioneer transcription factors access closed chromatin",
    "phase separation and transcriptional condensates at super-enhancers",
    "genomic imprinting and parent-of-origin specific expression",
]

# Method and assay questions -- "how do I measure X", answered by protocol and
# benchmarking papers.
METHODS = [
    "single-cell ATAC-seq protocols for profiling chromatin accessibility",
    "comparing Hi-C and Micro-C for detecting chromatin loops",
    "CUT and RUN versus ChIP-seq for transcription factor binding",
    "peak calling methods for ATAC-seq data and their trade-offs",
    "normalization methods for Hi-C contact matrices",
    "massively parallel reporter assays for measuring enhancer activity",
    "CRISPR screens for identifying functional regulatory elements",
    "benchmarking single-cell multiome joint RNA and ATAC profiling",
    "computational deconvolution of bulk chromatin accessibility data",
    "batch effect correction in single-cell genomics",
]

# Computational and modelling questions -- the ML-adjacent slice of the corpus.
COMPUTATIONAL = [
    "deep learning models that predict gene expression from DNA sequence",
    "transformer architectures applied to regulatory genomics",
    "predicting enhancer promoter interactions with machine learning",
    "interpreting convolutional neural networks trained on genomic sequence",
    "foundation models for single-cell transcriptomics",
    "graph neural networks for gene regulatory network inference",
    "predicting the effect of noncoding variants on gene expression",
    "evaluation metrics and pitfalls in genomic deep learning benchmarks",
]

# Disease and variant questions -- where regulation meets phenotype.
DISEASE = [
    "noncoding GWAS variants disrupting transcription factor binding sites",
    "enhancer hijacking in cancer and oncogene activation",
    "regulatory variants linked to autoimmune disease risk",
    "expression quantitative trait loci mapping in human tissues",
    "epigenetic silencing of tumour suppressor genes",
    "3D genome reorganisation in cancer cells",
    "structural variants that disrupt TAD boundaries and cause disease",
]

# Cell-type and developmental biology questions.
BIOLOGY = [
    "chromatin accessibility changes during stem cell differentiation",
    "cell type specific enhancers in the developing brain",
    "transcriptional regulation of immune cell activation",
    "lineage specifying transcription factors in haematopoiesis",
    "regulatory landscape of intestinal epithelial differentiation",
]

MANUAL_QUERIES: list[str] = [
    *MECHANISM,
    *METHODS,
    *COMPUTATIONAL,
    *DISEASE,
    *BIOLOGY,
]


def get_manual_queries(limit: int | None = None) -> list[str]:
    """The judging query set, optionally truncated.

    Truncation takes a prefix rather than a sample so that a smaller judging
    run is a strict subset of a larger one -- judgements already made stay
    valid if the set is later extended, instead of being stranded on queries
    that dropped out.
    """
    return MANUAL_QUERIES[:limit] if limit else list(MANUAL_QUERIES)
