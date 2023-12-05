import os


def create_folders_from_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return

    with open(file_path, 'r') as file:
        for line in file:
            # Splitting the line into parts and extracting the folder name
            _,_, folder_name = line.strip().split(' ', 2)
            folder_name = folder_name.replace('/', '@')
            folder_name = f"Automation/{folder_name}"
            print(folder_name)

            # Creating the folder if it doesn't exist
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print(f"Created folder: {folder_name}")
            else:
                print(f"Folder {folder_name} already exists.")


# Replace 'input.txt' with the path to your text file
create_folders_from_file('candidates.txt')