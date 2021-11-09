# Installation

To run `scan_for_kmers_motifs.R`:

step1: install all the dependencies:

```sh
$ conda install r-essentials bioconductor-jaspar2020 bioconductor-biostrings bioconductor-tfbstools bioconductor-motifmatchr bioconductor-summarizedexperiment r-doparallel bioconductor-rhdf5 bioconductor-hdf5array 
```

step2: run `Rscript scan_for_kmers_motifs.R -h`

e.g.,
```sh
$ Rscript scan_for_kmers_motifs.R -i peaks.bed -g hg19.fa -s 'Homo sapiens'
```
