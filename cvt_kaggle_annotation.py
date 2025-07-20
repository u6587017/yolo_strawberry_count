import os
import xml.etree.ElementTree as ET

xml_file = "kag_straw/annotations.xml"  
output_dir = "labels"         
os.makedirs(output_dir, exist_ok=True)

tree = ET.parse(xml_file)
root = tree.getroot()

for image in root.findall("image"):
    filename = os.path.basename(image.attrib["name"])
    width = float(image.attrib["width"])
    height = float(image.attrib["height"])
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, "w") as f:
        for box in image.findall("box"):
            label = box.attrib["label"]
            if label != "strawberry":
                continue
            class_id = 0
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            x_center = (xtl + xbr) / 2.0 / width
            y_center = (ytl + ybr) / 2.0 / height
            bbox_width = (xbr - xtl) / width
            bbox_height = (ybr - ytl) / height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
