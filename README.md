# ToxGet
A simple program to retrieve toxins with cysteine motifs from transcriptomes. Uses wavelet decomposition and cross correlation with a cysteine only embedding to accomplish this. If an unknown sequences scores over the threshold after crosscorrelation with one of the known toxin sequences, the unknown sequence is considered a toxin. Will also filter out sequences based on length and minimum cysteine count criteria.

## Requirements
Numpy >= 1.26.0 (any relatively recent version of numpy should suffice)

## Toxins
A file of 1,422 scorpion toxins, 89 calcium, 628 potassium, and 705 sodium channel toxins total to act as the comparison.

## Commands
+ `-ff` Fasta file or folder of fasta files for processing.
+ `-xp` Fasta file of known sequences for comparison.
+ `-of` Output folder path.
+ `-th` Threshold for crosscorrelation to consider two sequences to match.
+ `-ll` Lower bound for sequence length.
+ `-lu` Upper bound for sequence length.
+ `-mc` Minimum cysteine count required.

## Wavelet Transform
The filter banks to make the wavelet generating matrix used was provided by the paper 'Face Recognition Using M-Band Wavelet Analysis' by Mehri et al. (2012).
