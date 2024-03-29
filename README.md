# DeepMSP

### Setup

#### Step 1: Download Required Files

Download the following files required for the project:

- `clusters.csv`: A list of cluster IDs, e.g. "left_hemisphere.cluster_12345"
- `HCP_n1065_allDWI_fiber_clusters.csv`: A set of DWI measurements for each cluster of each subject
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
