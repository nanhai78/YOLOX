import os
import pickle
import xml.etree.ElementTree as ET

VOC_CLASSES = ['others']

def parse_rec2(filename):
    """Parse a PASCAL VOC xml file"""
    tree = ET.parse(filename)
    objects = []
    for root in tree.findall("object"):
        child = root.find("others")
        for obj in child.findall("others"):
            obj_struct = {}
            obj_struct["name"] = 'others'
            obj_struct["pose"] = 0
            obj_struct["truncated"] = 0
            obj_struct["difficult"] = 0
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ]
            objects.append(obj_struct)

    return objects

if __name__ == '__main__':
    res = parse_rec2("C:\\Users\\李刚\\Desktop\\2023-11-30-09-38-27_JY01_F_24.0_spr_5_0_5_0_1_5_6_2_6_2_3_6_25_1_1_R_B.xml")
    print(res.shape)