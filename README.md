# CNNeoPP: Consensus Neoantigen Prioritization Pipeline

CNNeoPP is a **rank-level consensus ensemble framework** for neoantigen prioritization. It integrates multiple sequence- and feature-based predictive models and prioritizes peptide–HLA pairs that are consistently supported across heterogeneous submodels, aiming to reduce false positives while retaining true positives **without relying on fixed cutoff thresholds**.

> Terminology note: **CNNeoPP** refers to the overall consensus prioritization framework in this repository. **CNNeo** refers to the immunogenicity prediction module used within CNNeoPP.

## Table of Contents

- [Overview](#overview)
- [Repository Contents](#repository-contents)
- [Environment and Installation](#environment-and-installation)
- [Input Data Format](#input-data-format)
- [Running CNNeoPP (End-to-End Workflow Reference)](#running-cnneopp-end-to-end-workflow-reference)
- [Output Description](#output-description)
- [Rank-Level Consensus Logic](#rank-level-consensus-logic)
- [Reproducibility and Versioning](#reproducibility-and-versioning)
- [Citation](#citation)
- [Contact](#contact)
- [Appendix A: CNNeoPP End-to-End Pipeline Commands](#appendix-a-cnneopp-end-to-end-pipeline-commands)

---

## Overview

CNNeoPP integrates multiple submodels to prioritize neoantigen candidates using **rank-level consensus**. Each submodel produces an independent ranked list of peptide–HLA candidates. CNNeoPP then promotes candidates that are consistently supported across models, reducing false positives while retaining true positives without requiring fixed score cutoffs.

## Repository Contents

This repository contains:

- `models/`  
  Pretrained model weights used in the manuscript

- `data/example/`  
  Minimal example input and expected output files

- `docs/`  
  Additional documentation describing the CNNeoPP workflow

---

## Environment and Installation

This repository uses **`environment.yml`** as the primary (and recommended) way to reproduce the runtime environment.

### Prerequisites

- Install **Miniconda** or **Anaconda** (Conda is required to use `environment.yml`).
- Recommended: use the same OS family as the environment was exported on.  
  The provided `environment.yml` was exported from a Windows Anaconda environment and may require small adjustments on other platforms.

### Create the environment (recommended)

From the repository root (where `environment.yml` is located):

```bash
# Create a new environment named "cnneopp" from environment.yml
# (Recommended to avoid conflicts if the yml has name: base)
conda env create -f environment.yml -n cnneopp

conda activate cnneopp
```

If you already created the environment once and want to update it:

```bash
conda env update -n cnneopp -f environment.yml --prune
conda activate cnneopp
```

### Verify installation

```bash
python --version
python -c "import torch, transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__)"
```

### Optional: enable Jupyter kernel

If you run notebooks, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name cnneopp --display-name "Python (cnneopp)"
```

### GPU/CPU note (PyTorch)

The exported `environment.yml` pins PyTorch wheels that include a CUDA build (`+cu121`). If you are using a CPU-only machine or a different CUDA setup, environment creation may fail during the pip stage, or PyTorch may not run as expected.

A robust approach is:

1. Create the environment from `environment.yml`.
2. If needed, reinstall PyTorch to match your hardware.

Example (CPU-only):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Example (CUDA 12.1):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Input Data Format

CNNeoPP operates on a tabular input file (**CSV** or **TSV**) containing neoantigen candidate information:

- `peptide` — Amino-acid sequence of the candidate neoantigen  
- `hla` — HLA allele (e.g., `HLA-A*02:01`)

Optional feature columns (if available):

- Proteasomal cleavage
- TAP transport efficiency
- NetCTLpan score
- Peptide–HLA binding affinity
- Peptide–HLA binding stability
- Peptide hydrophobicity
- Peptide weight
- Peptide entropy
- IEDB immunogenicity score
- Agretopicity
- T-cell contact residues hydrophobicity

Example input file:

- `data/example/input_features.csv`

---

## Running CNNeoPP (End-to-End Workflow Reference)

The end-to-end workflow covers:

1. Submodel inference  
2. Rank-level consensus integration  
3. Final peptide ranking  

Appendix A provides a reference pipeline for common upstream steps, followed by immunogenicity/neoantigen prioritization.

---

## Output Description

The primary output is a ranked list of peptide–HLA pairs, typically including:

- `final_rank`
- `consensus_support`
- `model_scores`

Example output file:

- `data/example/example_output.csv`

---

## Rank-Level Consensus Logic

Each submodel independently produces a ranked list of peptide–HLA pairs. The final CNNeoPP ranking is generated by:

1. Selecting the **Top-N** candidates from each submodel  
2. Identifying peptides supported by **at least two** submodels  
3. Prioritizing consensus peptides at the top of the final ranking  
4. Appending remaining peptides based on individual model scores  

---

## Reproducibility and Versioning

- Use the committed `environment.yml` to recreate the environment: `conda env create -f environment.yml -n cnneopp`.
- Record the following when reporting results: OS, Python version, CUDA availability (if applicable), and the exact Git commit/tag.
- For strict reproduction of published results, use a tagged release corresponding to the manuscript version.

---

## Citation

Please cite CNNeoPP as:

> Yu Cai, Rui Chen, Mingming Song, Lei Wang, Zirong Huo, Dongyan Yang, Sitong Zhang, Shenghan Gao, Seungyong Hwang, Ling Bai, Yonggang Lv, Yali Cui, Xi Zhang. **CNNeoPP: A large language model-enhanced deep learning pipeline for personalized neoantigen prediction and liquid biopsy applications**.

---

## Contact

For questions or issues, please open a GitHub issue or contact the authors.

---

# Appendix A: CNNeoPP End-to-End Pipeline Commands

> All commands below are provided as a reference template. Replace placeholder paths (e.g., `file path/`) and names (e.g., `file name`) with your actual file locations and sample identifiers.

## 1. Quality control (fastqc)

```bash
file path/fastqc file path/file name.fastq.gz -o file path/file name
```

## 2. Remove sequencing adapter sequences and low-quality reads (Trimmomatic)

```bash
java -jar file path/trimmomatic-0.39.jar PE   -phred33   -threads 30   file path/file name1.fq.gz file name2.fq.gz   file path/output_file name1paired.fastq.gz output_file name1unpaired.fastq.gz   file path/output_file name2paired.fastq.gz output_file name2unpaired.fastq.gz   ILLUMINACLIP:file path/TruSeq3-PE.fa:2:30:10   SLIDINGWINDOW:5:20   LEADING:5   TRAILING:5   MINLEN:50
```

## 3. Alignment (BWA)

```bash
bwa index -a bwtsw Homo_sapiens_assembly38.fasta

bwa mem -t 6 -M   -R '@RG\tID:foo_lane\tPL:illumina\tLB:library\tSM:GAJ_Z'   file path/Homo_sapiens_assembly38.fasta.64   output_file name1paired.fastq.gz output_file name2paired.fastq.gz   | samtools view -S -b - > file path/file name.bam
```

## 4. Sort (samtools)

```bash
time samtools sort -@ 4 -m 4G -O bam -o file name.sorted.bam file name.bam
```

## 5. Mark and remove PCR duplicates (Picard)

```bash
java -jar file path/picard.jar MarkDuplicates   -REMOVE_DUPLICATES true   -I file path/file name.sorted.bam   -O file name.sorted.markdup.bam   -M file name.markdup_metrics.txt

samtools index file name.sorted.markdup.bam
```

## 6. Base quality recalibration (GATK4.4)

```bash
gatk BaseRecalibrator   -R file path/Homo_sapiens_assembly38.fasta   -I file path/file name.sorted.markdup.bam   --known-sites file path/Homo_sapiens_assembly38.dbsnp138.vcf   --known-sites file path/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz   --known-sites file path/1000G_phase1.snps.high_confidence.hg38.vcf.gz   -O file name_recal_data.table

gatk ApplyBQSR   -R file path/Homo_sapiens_assembly38.fasta   -I file path/file name.sorted.markdup.bam   --bqsr-recal-file file name_recal_data.table   -O file name_recalibrated.bam
```

## 7. Estimate contamination (GATK4.4)

```bash
gatk GetPileupSummaries   -I normal file name_recalibrated.bam   -V file path/small_exac_common_3.hg38.vcf.gz   -L file path/small_exac_common_3.hg38.vcf.gz   -O file path/normal file name_getpileupsummaries.table

gatk GetPileupSummaries   -I tumor file name_recalibrated.bam   -V file path/small_exac_common_3.hg38.vcf.gz   -L file path/small_exac_common_3.hg38.vcf.gz   -O file path/tumor file name_getpileupsummaries.table

gatk CalculateContamination   -I tumor file name_getpileupsummaries.table   -matched normal file name_getpileupsummaries.table   -tumor-segmentation segments.table   -O file name_calculatecontamination.table
```

## 8. Somatic variant calling (Mutect2)

```bash
gatk Mutect2   -R file path/Homo_sapiens_assembly38.fasta   -I file path/tumor file name_recalibrated.bam -tumor tumor file name   -I file path/normal file name_recalibrated.bam -normal normal file name   -pon file path/1000g_pon.hg38.vcf.gz   --germline-resource file path/af-only-gnomad.hg38.vcf.gz   -L file path/wgs_calling_regions.hg38.interval_list   -O file name_somatic.vcf.gz   -bamout file name_tumor_normal.bam
```

## 9. Mutation filtering (GATK 4.4)

```bash
gatk FilterMutectCalls   -R file path/Homo_sapiens_assembly38.fasta   -V file path/file name_somatic.vcf.gz   --contamination-table file path/file name_calculatecontamination.table   --stats file path/file name_somatic.vcf.gz.stats   --tumor-segmentation file path/segments.table   -O file name_somatic_oncefiltered.vcf.gz
```

## 10. Keep only PASS (GATK 4.4)

```bash
gatk SelectVariants   -R file path/Homo_sapiens_assembly38.fasta   -V file path/file name_somatic_oncefiltered.vcf.gz   -select "vc.isNotFiltered()"   -O file path/file name_somatic_oncefiltered_SelectVariants-filtered.vcf
```

## 11. Annotation (Annovar)

```bash
time file path/table_annovar.pl   file path/file name_somatic_oncefiltered_SelectVariants-filtered.vcf   file name/annovar/humandb/   -buildver hg38   -out file path/file name   -remove   -protocol refGene,cytoBand,exac03,avsnp147,dbnsfp30a   -operation gx,r,f,f,f   -nastring .   -vcfinput   -polish   && echo "** annotation done **"
```

## 12. Gene expression (Kallisto)

```bash
kallisto index -i kallisto_index file name/Homo_sapiens.GRCh38.cdna.all.fa
kallisto quant -i kallisto_index -o file_name file_name/file name1_cutadapted.fastq.gz file_name/file name2_cutadapted.fastq.gz
```

## 13. HLA typing (Optitype)

```bash
razers3 -i 95 -m 1 -dr 0 -o file path/fished_1.bam file path/hla_reference_dna.fasta output_file name_paired.fastq
samtools bam2fq fished_1.bam > output_file name_paired_fished.fastq
python OptiTypePipeline.py -i sample_fished_1.fastq sample_fished_2.fastq --dna -v -o output_dir
```

## 14. Binding affinity (NetMHCpan-4.1)

NetMHCpan-4.1 service:

- https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/

## 15. Immunogenicity prediction using CNNeo

The corresponding code is provided in the `CNNeo_model` directory.
