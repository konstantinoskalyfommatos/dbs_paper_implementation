# Project Pipeline

## Preliminaries

1. **Set Up Directories**:  
   Creates the directories to save plots, models and data:

  ```bash
  bash src/pipelines/preliminaries.sh
  ```


2. **Docker Setup**:  
    ```bash
      docker build -t dbs_paper_implementation .
    ```

---

## Pipeline Execution

Run the following scripts in order:

1. **Data Processing Pipeline**:  
Preprocesses the raw data and prepares it for training.  

    ```bash
      bash  ./src/pipelines/data_processing_pipeline.sh
    ```

2. **Training Pipelinee**:  
Trains the model using the processed data.

    ```bash
      bash ./src/pipelines/train_pipeline.sh
    ```

3. **Fine-Tuning Gemma**:  
- Open the notebook `finetune_gemma.ipynb`.
- Execute all cells sequentially to fine-tune the Gemma model.

4. **Testing Pipelines**:
- **Ground Truth Testing**: Evaluates the model against ground truth labels.

  ```bash
    bash ./src/pipelines/test_pipeline_ground_truth.sh
  ```

- **End-to-End Testing**: Runs the full pipeline, including model inference and evaluation.

  ```bash
    bash ./src/pipelines/test_pipeline_e2e.sh
  ```

