from ExperimentalPipeline1 import main

# Specific rows correspond to configurations defined in the CSV files (See ExperimentList folder).
if __name__ == '__main__':
    specific_rows = [206]
    Mode = "tox21"
    main(specific_rows, Mode)

