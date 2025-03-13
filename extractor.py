import xml.etree.ElementTree as ET
import os
import sys
import shutil
sys.stdout.reconfigure(encoding='utf-8')
# mendley dataset grape path
file_path = 'path_to_your_xml_file.xml'

def read_xml(directory, filename):
    try:
        xml_filename = filename + '.xml'
        jpg_filename = filename + '.jpg'
        tree = ET.parse(directory + xml_filename)
        root = tree.getroot()
        folder= root.find('folder').text
        filename_by_xml= root.find('filename').text
        simple_path = directory + jpg_filename
        if os.path.exists(simple_path):
            if folder == '' or folder is None:
                folder = 'unknown'
            sub_dir = directory + folder + '/'
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            # copy file to sub directory
            shutil.copy(simple_path, sub_dir + jpg_filename)

            return True
        return False


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
    directory = 'd:/microbify/weinreebe/labeled/'
    found=0
    not_found=0
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            filename = os.path.splitext(filename)[0]
            if read_xml(directory, filename):
                found+=1
            else:
                not_found+=1
    print(f"Found: {found}, Not found: {not_found}")
            # break