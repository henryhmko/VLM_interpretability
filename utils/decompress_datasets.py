import os
import zipfile
import argparse

def decompress_zip(zip_path, output_directory):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(output_directory)
    print("all done")
    print(output_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Decompresses image dataset zipfile to a single folder. Made for ease of sharing small datasets."
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type = str,
        help = "Input path to zipfile"
    )

    args = parser.parse_args()

    # Path to zipfile
    path_to_dir_above = os.path.dirname(os.getcwd())
    output_directory = os.path.join(path_to_dir_above, 'data')

    dataset_zipfile_name = os.path.basename(args.input_path)
    dataset_name = os.path.splitext(dataset_zipfile_name)[0]
    
    output_path = os.path.join(output_directory, dataset_name)

    decompress_zip(args.input_path, output_path)