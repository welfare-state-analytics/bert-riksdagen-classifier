import argparse
import csv
import os
import re


def clean_content(content):
	# Remove line breaks and trailing '\n'
	content = str(content).strip('n')
	# Remove quotation marks
	content = content.replace('"', '')

	# Remove symbols
	content = re.sub(r'[^\w\s()]', '', content)

	return content


def combine_csv_files(directory):
	
	# Extract tag value from folder name
	tag_value = os.path.basename(os.path.normpath(directory))
	# Output CSV file name
	output_file = os.path.join('./data/pilot', f'{tag_value}-combined.csv')

	# Columns to extract from each CSV file
	columns_to_extract = ['content', 'tag']

	# Initialize list to store combined rows
	combined_rows = []

	# Loop through each CSV file in the directory
	for filename in os.listdir(directory):
		if filename.endswith('.csv'):
			filepath = os.path.join(directory, filename)
			with open(filepath, 'r', newline='') as csvfile:
				reader = csv.DictReader(csvfile)
				for row in reader:
					# Modify the 'tag' value
					row['tag'] = tag_value
					 # Remove line breaks and double spaces from content
					row['content'] = clean_content(row['content'])
					combined_row = {column: row[column] for column in columns_to_extract}
					combined_rows.append(combined_row)

	# Write combined rows to the output CSV file
	with open(output_file, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=columns_to_extract)
		writer.writeheader()
		writer.writerows(combined_rows)

	print("Combined CSV file created:", output_file)



def merge_combined_csv_files(folder_path):
	# Initialize list to store combined rows
	combined_rows = []
	

	# Output file path
	output_file = folder_path + '/all-data.csv'
	# Get all files in the folder that contain "combined" in their filename
	combined_files = [file for file in os.listdir(folder_path) if "combined" in file]

	# Loop through each combined file
	for combined_file in combined_files:
		with open(os.path.join(folder_path, combined_file), 'r', newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			# Skip the header if not the first file
			if combined_files.index(combined_file) != 0:
				next(reader)
			for row in reader:
				combined_rows.append(row)

	# Write combined rows to the output CSV file
	with open(output_file, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=reader.fieldnames)
		writer.writeheader()
		writer.writerows(combined_rows)

	print("CSV files merged into:", output_file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Combine CSV files from a specified directory.")
	parser.add_argument("--dir", help="Path to the directory containing CSV files.")
	args = parser.parse_args()

	#combine_csv_files(args.dir)
	merge_combined_csv_files(args.dir)
