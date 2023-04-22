import os
import xml.etree.ElementTree as ET

def parse_rec(filename):
    """Parse a PASCAL VOC xml file"""
    tree = ET.parse(filename)
    objects = []
    for obj_1 in tree.findall("object"):  # 第一级目录
        # if obj.find("name").text not in VOC_CLASSES:
        #     continue
        for obj_2 in obj_1.findall('*'):  # 第二级目录
            for obj_3 in obj_2.findall('*'):  # 第三级目录
                obj_struct = {}
                obj_struct["name"] = "others"  # 节点名就是目标的名称
                obj_struct["pose"] = 'Unspecified'
                obj_struct["truncated"] = 0
                obj_struct["difficult"] = 0
                bbox = obj_3.find("bndbox")
                obj_struct["bbox"] = [
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text),
                ]
                objects.append(obj_struct)
    return objects


if __name__ == '__main__':
    xml_path = "D:\\python_all\\Workspace002\\new_xmlfile"
    for filename in os.listdir(xml_path):
        objects = parse_rec(os.path.join(xml_path, filename))
        print(objects)