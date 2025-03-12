import xml.etree.ElementTree as ET
import os
import sys
import shutil
sys.stdout.reconfigure(encoding='utf-8')
# mendley dataset grape path
file_path = 'path_to_your_xml_file.xml'

def read_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        folder_elem = root.find('folder')
        filename_elem = root.find('filename')
        path_elem = root.find('path')

        folder = folder_elem.text if folder_elem is not None else None
        filename = filename_elem.text if filename_elem is not None else None
        path = path_elem.text if path_elem is not None else None

        # print("Folder:", folder)
        # print("Filename:", filename)
        # print("Path:", path)
        imgpath=os.path.dirname(file_path)
        imgpath=os.path.join(imgpath, filename)
        if not os.path.exists(imgpath):
            print("Image path does not exist:", imgpath)
        parent_dir = os.path.dirname(file_path)
        parent_of_parent_dir = os.path.dirname(parent_dir)
        # print(f'parent_of_parent_dir: {parent_of_parent_dir}')
        target_folder = os.path.join(parent_of_parent_dir, folder)
        target= os.path.join(target_folder, filename)
        source_image = imgpath
        if os.path.exists(source_image):
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy(source_image, target)
            # print(f"Copied {source_image} to {target}")
        else:
            print(f'Image does not exist: {source_image}')
            print(f'parent_of_parent_dir: {path}')
        # else:
        #     print("Image path exists:", imgpath)

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def sort_images(file_path):
  pass  

# Example usage
if __name__ == "__main__":
    # list all xml files at the path c:/data/grape/
    directory = 'c:/data/grape/'
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            read_xml(file_path)
            # break