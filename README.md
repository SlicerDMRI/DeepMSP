# DeepMSP

### Setup

#### Step 1: Download Required Files

Download the following files required for the project. See 'File Structure Details/Examples' for more details:

- `clusters.csv`: A list of fiber cluster IDs, e.g. "left_hemisphere.cluster_12345".
- `HCP_n1065_allDWI_fiber_clusters.csv`: A set of DWI measurements for each cluster of each subject.
- `S1200_demographics_Behavioral.csv`: A set of individual functional performance measures for each subject

#### Step 2: Create Dataset

Run the following command to create the dataset needed for training:

```bash
python create_dataset.py
```

#### Step 3: Generate CSV Files

To generate the .csv files needed for the project, execute:

```bash
python generate_csvs.py
```

### Usage

#### Training the Model

To train the model, run train_multi.py with your desired arguments, for example:

```bash
python train_multi.py --batch_size 10 --epochs 50 --learning_rate 1e-4 --input_channels 1940 --results_dir ./results --dataset_dir new_dataset --dropout 10 --save_name cerebellum_optimised_transformer
```

#### Performing Clustering

After training, perform parcellation by running explain_multi.py with your desired arguments, for example:

```bash
python explain_multi.py --save_name clustering_results --results_name cerebellum_optimised_transformer --bilateral
```

### File Structure Details/Examples

#### clusters.csv

A simple headerless .csv file where each row contains a single cluster ID string.

|                                |
|--------------------------------|
| left_hemisphere.cluster_12345  |
| right_hemisphere.cluster_12345 |
| commissural.cluster_54321      |
| ...                            |

#### HCP_n1065_allDWI_fiber_clusters.csv

Each row contains a subject ID followed by a variety of DWI measures for each cluster ID of that subject. The create_dataset.py script assumes this .csv file contains columns for the Min/Max/Median/Mean/Variance for: FA1, FA2, Num_Fibers, Num_Points; Min/Max/Median/Mean for: correct_trace1, correct_trace2; and additionally: Num_Fibers, Num_Points columns.

| subjectkey | left_hemisphere.cluster_12345.Num_Fibers | left_hemisphere.cluster_12345.FA1.Mean | ... | right_hemisphere.cluster_12345.NumFibers | ... |
|:----------:|:----------------------------------------:|:--------------------------------------:|:---:|:----------------------------------------:|:---:|
|   555555   |                    304                   |                  0.34                  | ... |                    403                   | ... |
|   333333   |                    201                   |                  0.55                  | ... |                    204                   | ... |
|   444444   |                    111                   |                  0.32                  | ... |                    112                   | ... |
|     ...    |                    ...                   |                   ...                  | ... |                    ...                   | ... |

#### S1200_demographics_Behavioral.csv

Each row contains a set of functional/behavioral measures for the corresponding subject. If you are using functional/behavioral measures that are not used by the HCP-YA dataset, you may need to alter the names of the headers in the generate_csvs.py file to match your dataset's headers.

| Subject | PicVocab_AgeAdj | ReadEng_AgeAdj | ... |
|:-------:|:---------------:|:--------------:|:---:|
|  555555 |       102       |       85       | ... |
|  333333 |        99       |       94       | ... |
|  444444 |       105       |       83       | ... |
|   ...   |       ...       |       ...      | ... |
