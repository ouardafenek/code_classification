**The Notebook** 
  
  It contains:
    * The pre-processing of the dataset -> Transformation to DataFrame (from code_classification_dataset to dataset_raw_parquet) 

    * The tags filtering: Droping the rows of the dataframe that don't have any of the tags: "math", "graphs",
    "strings", "number theory", "trees", "geometry", "games", "probabilities". 

    * The MultiHot Encoding 

    * The creation of 3 new columns in the df: cleaned_source_code, cleaned_prob_desc_description, cleaned_describtion_and_source_code 

    * The Exploratory Data Analyses: 
        - Tags frequency 
        - Tags correlation 
        - Sequences lengths 
        - Most frequent words in the corpus (after cleaning)


    * Dataset split : Using statisfaction to deal with unbalenced labels




**TRAIN COMMANDS** 

**Usage**: 
    train_baseline_cli.py [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH]
                             [--text_column TEXT_COLUMN] [--preprocess {none,nltk}]





    train_chains_cli.py [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH]
                           [--text_column TEXT_COLUMN] [--n_chains N_CHAINS]
                           [--max_features MAX_FEATURES] [--threshold THRESHOLD]





    train_codeBERT_cli.py [-h] [--input_path INPUT_PATH] [--bs BS] [--lr LR] [--epochs EPOCHS]
                             [--output_path OUTPUT_PATH] [--threshold THRESHOLD]

            Train CodeBERT for multilabel classification on code + text data.

            options:
              -h, --help            show this help message and exit
              --input_path INPUT_PATH
                                    Path to the input directory containing the train/test splits (X_train.parquet,
                                    y_train.parquet, X_test.parquet, y_test.parquet). Default:
                                    ../data/code_classification_split
              --bs BS               Batch size used for training and evaluation. Larger values use more GPU memory.
                                    Default: 8
              --lr LR               Learning rate for the AdamW optimizer. Typical values for CodeBERT are in the
                                    range [1e-5, 5e-5]. Default: 2e-05
              --epochs EPOCHS       Number of training epochs. Increasing this may improve performance but can lead
                                    to overfitting. Default: 30
              --output_path OUTPUT_PATH
                                    Directory where the fine-tuned model, tokenizer, and evaluation plots will be
                                    saved. Default: ../outputs/multilabel_CodeBERT
              --threshold THRESHOLD
                                    Decision threshold applied to sigmoid outputs during evaluation to convert
                                    probabilities into binary predictions. Must be in the range [0, 1]. Default:
                                    0.5



**PREDICT ONE SAMPLE** 

**Usage**: 

    predict_cli.py [-h] [--model_type {codebert,chain}] [--model_path MODEL_PATH] --input INPUT [--threshold THRESHOLD]

**Examples**:

    ./predict_cli.py --input ../data/code_classification_dataset/sample_0.json --model_type chain
    ./predict_cli.py --input ../data/code_classification_dataset/sample_0.json --model_type codebert




**EVALUATE RESULTS ON A FOLDER** 

**Usage**: 

    evaluate_cli.py [-h] [--model_type {codebert,chain}] [--model_path MODEL_PATH] --input_folder INPUT_FOLDER [--threshold THRESHOLD]

**Examples**:

    ./evaluate_cli.py --input_folder ../data/code_classification_dataset_small --model_type chain
    ./evaluate_cli.py --input_folder ../data/code_classification_dataset_small --model_type codebert