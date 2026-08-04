# Time-Aware Combination of Zero-Shot Proxies and Successive Halving for Unseen Datasets

Submission for the NAS Unseen-Data Competition 2026 </br>

>This project implements a time-aware Neural Architecture Search pipeline for image classification on previously unseen datasets. Since dataset properties, task difficulty, and available runtime may vary between evaluations, the complete pipeline adapts its preprocessing, architecture search, and final training procedure to the provided data and remaining competition time. </br>
>To find out more information, including dates and rules, please visit the competition website: [https://www.nascompetition.com](https://www.nascompetition.com).

Our approach searches a compact cell-based convolutional architecture space. Randomly generated candidates are ranked using a weighted combination of the zero-shot proxies NASWOT and SynFlow. Only the highest-ranked candidates are evaluated with Hyperband and Successive Halving, allowing unpromising architectures to be removed after small training budgets while stronger candidates receive additional epochs. 

The available runtime is estimated directly on the current dataset and hardware. The search budget, Hyperband epoch budgets, number of search passes, final training duration and prediction safety margin are adjusted using the competition clock.
## Project Structure

```text
.
├── datasets/                  # Local datasets
│   ├── <dataset_name>/        # Here dataset A
│   └── <dataset_name>/        # Here dataset B
│ 
├── evaluation/                # Provided evaluation pipeline 
│   ├── main.py                # Main-entry point
│   └── score.py               # Computes the final score
│ 
├── submission_final/          # Our competition submission
│   ├── data_processor.py      # Dataset preprocessing and DataLoader creation
│   ├── helpers.py             # Shared helper functions
│   ├── nas.py                 # Search space, zero-shot filtering and successive halving
│   └── trainer.py             # Final model training
│ 
├── Makefile                   # Provided build, test, and packaging commands   
├── README.md                  # This file
├── README_Project.md          # Original competition documentation
├── requirements.txt           # Python dependencies 
├── submission_final_log1.log  # 1-hour evaluation run on Conway, Cryptic, and Windspeed
└── submission_final_log2.txt  # 1-hour evaluation run on Conway, Cryptic, and Windspeed
```
Following are the files we worked on, the other files were provided with the template.

### `data_processor.py`
Prepares the raw NumPy data for the search and training pipeline. It converts grayscale inputs to channel-first tensors, optionally resizes large square images, applies dataset-specific normalization, and selects the batch size based on image resolution and dataset size.

### `nas.py`
Implements our time-aware architecture search. Candidate cell architectures are pre-filtered and ranked using the zero-shot proxies NASWOT and SynFlow.
The most promising candidates are evaluated with Hyperband and Successive Halving.

### `trainer.py`
Continues training the best architecture found during the search with a strict time budget.

## Quick Start
Install the dependencies from requirements.txt:
```text
python -m pip install --upgrade pip 
python -m pip install -r requirements.txt
```

To run our submission use the following command. </br>
This command executes the complete competition pipeline:
```bash
make submission=submission_final all
```

### Dataset Setup
Create a `datasets` directory if not available.

Each dataset must follow the structure expected by the competition evaluation pipeline. It must contain the NumPy arrays for the training, validation, and test splits, together with a metadata file.

>Example metadata for AddNIST with a time limit of 1 hour:<br>
>`{"num_classes": 20, "input_shape": [50000, 3, 28, 28], "codename": "Adaline", "benchmark":89.850, "time_limit": 1.0} `

The `time_limit` value controls the dataset-specific runtime in hours.

>The available and recommended datasets can be found in the `README_Project.md` under the `Datasets` section

