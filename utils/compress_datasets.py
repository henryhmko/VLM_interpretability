import os
import zipfile
import argparse

def compress_directory(directory_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Compresses image dataset folder into a single zipfile. Made for ease of sharing small datasets."
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type = str,
        help = "Input path of dataset"
    )

    args = parser.parse_args()

# Directory path containing the files and subdirectories with images
output_path = os.path.join(os.getcwd(), 'compressed_datasets')

# Output ZIP file path
dataset_name = os.path.basename(args.input_path)
output_zipfile_name = f"{dataset_name}.zip"

output_zipfile_path = os.path.join(output_path, output_zipfile_name)

# Compress the directory
compress_directory(args.input_path, output_zipfile_path)
print(f'Compression complete. ZIP file created: {output_zipfile_path}')