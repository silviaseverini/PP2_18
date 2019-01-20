# PP2_18


The folder "checkpoints" contains the trained model.

To run inference:

	python prediction.py -i input_file_to_be_predicted.txt -o output_file_to_be_generated.txt

Example:

	python prediction.py -i ./dataset/testing.fasta -o ./dataset/testing_result.fasta