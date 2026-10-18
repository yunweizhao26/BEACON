#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
if (!requireNamespace("kSamples", quietly = TRUE)) {
    stop("kSamples is required in the auditor's R environment; no package is installed automatically")
}
destination <- args[[1L]]
provenance <- sub("[.]csv$", ".manifest.txt", destination)
stopifnot(!file.exists(destination), !file.exists(provenance))
grid <- seq(-5, 1000, by = 0.001)
p <- kSamples::ad.pval(grid, m = 1, version = 1)
stopifnot(all(is.finite(p)), all(p >= 0 & p <= 1), all(diff(p) <= 1e-14))
p <- pmax(p, .Machine$double.xmin)
options(digits = 17)
write.table(data.frame(statistic = grid, pvalue = p), file = destination,
            sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)
probes <- seq(-1.2, 100, length.out = 997) + 0.00037
direct <- kSamples::ad.pval(probes, m = 1, version = 1)
interpolated <- approx(grid, p, xout = probes)$y
stopifnot(max(abs(interpolated - direct)) < 1e-5)
writeLines(c(paste("R:", getRversion()), paste("kSamples:", packageVersion("kSamples")),
             "Function: ad.pval(T, m=1, version=1)",
             "Grid: [-5,1000], step 0.001; interpolation in P; underflow floor double.xmin",
             paste("Withheld-point maximum absolute error:", max(abs(interpolated - direct))),
             "This reconstructs the published method; the authors' original interpolation grid was not supplied.",
             paste("Job ID:", Sys.getenv("SLURM_JOB_ID")),
             paste("CPU/wall:", paste(proc.time(), collapse = " ")),
             grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE),
             capture.output(sessionInfo())), provenance)
