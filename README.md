# LevenshteinDistance:

The project was implemented as a part of the _Graphic Processors in Computational Applications_ course at Warsaw University of Technology during the winter semester of the 2024-2025 academic year.

[Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) is a measure of the similarity between two strings, which takes into account the number of insertion (`i`), deletion (`d`) and substitution (`s`) operations needed to transform one string into the other.

![lev function](Images/formula.png)

The project contains two different implementations of the algorithm:

- `cpu` is an implementation that uses dynamic programming and runs entirely on the CPU.
- `gpu` is an implementation that utilizes custom functions executed on the graphics card. The parallel version of the algorithm adapted for CUDA environments is based on the following [research paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186251).


## Data Specifications:
- $m$ - length of the source word.
- $n$ - length of the target word.

## Input Data Format:

Two input data formats are implemented in the program.

### Text Format:

The first line of the file contains two natural numbers, separated by whitespace.
These numbers are interpreted as $m$ and $n$. The second line of the file contains a string of length $m$, interpreted as the source word. 
The third line of the file contains a string of length $n$, interpreted as the target word.

For example, for $m = 8$, $n = 6$, the input file looks like this:

```c
8 6
saturday
sunday
```

### Binary Format:

This format is similar to the text format. At the beginning of the file are the parameters $m$ and $n$, followed by the data interpreted as the source and target words, respectively.

## Output Data Format:

The results are saved in text format only. 
The first line contains the edit distance between the input words, and the second line contains a string of operations that transform the source word into the target word.

For example, for $m = 8$, $n = 6$, the output file looks like this:

```c
8
ssd-isss
```

## Running the Program:

```c
LevenshteinDistance data_format computation_method input_file output_file
```

The program takes 4 positional parameters:

- `data_format`, which specifies the input data format (`txt`|`bin`)
- `computation_method`, which specifies the algorithm to use (`cpu`|`gpu`)
- `input_file`, which specifies the path to the input file in the appropriate format
- `output_file`, which specifies the path to the output file
  - if the file does not exist, it will be created
  - if the file exists, its contents will be overwritten by the current run of the program

Additionally, `LevenshteinDistance.Scripts` contains the script `levenshtein_generate_data.py`, which can generate sample input files in both allowed formats.
