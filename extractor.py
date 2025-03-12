import xml.etree.ElementTree as ET

# mendley dataset grape path
file_path = 'path_to_your_xml_file.xml'

def read_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
def example():
    root = read_xml(file_path)
    if root is not None:
        for child in root:
            print(child.tag, child.attrib)
def categorize():
    print("Categorizing...")
# Example usage
if __name__ == "__main__":
    categorize()