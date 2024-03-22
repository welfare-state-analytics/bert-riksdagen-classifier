import pandas as pd
import argparse
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



def update_tag(tag):
	# Update tag value
	tag = str(tag)
	if tag.lower() == 'u':
		return 'seg'
	elif tag.lower() not in ['u', 'seg']:
		return 'note'
	return tag.lower()

def process_csv_files(usage, input_dir, output_dir):
	# List to store dataframes from each file
	dfs = []

	# Loop through files in the directory
	for filename in os.listdir(input_dir):
		if filename.endswith('.csv') and usage in filename:
			print (f'{usage} files:', filename)
			filepath = os.path.join(input_dir, filename)
			# Read CSV file into a pandas dataframe
			df = pd.read_csv(filepath)
			# Append the dataframe to the list
			dfs.append(df)

	# Concatenate all dataframes into one
	combined_df = pd.concat(dfs, ignore_index=True)

	# Extract specific columns
	selected_columns = ['full_text', 'segmentation', 'github', 'protocol_id']
	selected_data = combined_df[selected_columns]

	# Rename columns
	selected_data.rename(columns={'full_text': 'content', 'segmentation': 'tag'}, inplace=True)
	

	selected_data['content'] = selected_data['content'].apply(clean_content)
	selected_data['tag'] = selected_data['tag'].apply(update_tag)
	# Save to a new CSV file
	output_file = os.path.join(output_dir,f'{usage}_full_text_merged_intro_note.csv')
	selected_data.to_csv(output_file, index=False)
	print(f"Data has been saved to {output_file}")

if __name__ == "__main__":
	# Create argument parser
	parser = argparse.ArgumentParser(description='Process CSV files.')
	parser.add_argument('--usage', type=str, help='train or val')
	parser.add_argument('--input_dir', type=str, help='Input directory containing CSV files')
	parser.add_argument('--output_dir', type=str, help='Output directory to save processed data')
	

	# Parse arguments
	args = parser.parse_args()

	# Call the function with input and output directories
	process_csv_files(args.usage, args.input_dir, args.output_dir)
