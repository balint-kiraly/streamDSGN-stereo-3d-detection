import os

def combine_files(input_dir, output_file):
    """Reads all files from a directory and combines their content into a new file."""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in sorted(os.listdir(input_dir)):
            file_path = os.path.join(input_dir, filename)

            if os.path.isfile(file_path):  # Ensure it's a file, not a directory
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())

if __name__ == "__main__":
    input_directory = "data/kitti_tracking/training/label_02"
    output_filename = "data/kitti_tracking/training/label_02/0000.txt"

    combine_files(input_directory, output_filename)
    print(f"Files combined into {output_filename}")