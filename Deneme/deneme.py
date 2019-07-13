import os, random
import xml.etree.ElementTree as ET

classes = ["RBC", "WBC", "Platelets"]
ratio = 0.9


def gen_det_rec(classes, ratio=1):
    assert ratio <= 1 and ratio >= 0
    img_dir = r"C:\Users\BULUT\Desktop\BCCD_Dataset-master\BCCD\JPEGImages"
    label_dir = r"C:\Users\BULUT\Desktop\BCCD_Dataset-master\BCCD\Annotations"
    img_names = os.listdir(img_dir)
    img_names.sort()
    label_names = os.listdir(label_dir)
    label_names.sort()
    file_num = len(img_names)
    assert file_num == len(label_names)

    idx_random = list(range(file_num))
    random.shuffle(idx_random)

    with open(r"C:\Users\BULUT\Desktop\boodcelltrain.txt", "w") as train_lst:
        print("Writing in train.lst...")
        for idx in range(file_num):
            each_img_path = os.path.join(img_dir, img_names[idx])
            each_label_path = os.path.join(label_dir, label_names[idx])
            tree = ET.parse(each_label_path)
            root = tree.getroot()
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)

            for obj in root.iter('object'):
                line = ""
                line += each_img_path
                cls_name = obj.find('name').text
                if cls_name not in classes:
                    continue
                cls_id = classes.index(cls_name)
                xml_box = obj.find('bndbox')
                xmin = float(xml_box.find('xmin').text)
                ymin = float(xml_box.find('ymin').text)
                xmax = float(xml_box.find('xmax').text)
                ymax = float(xml_box.find('ymax').text)
                for i in [xmin, ymin, xmax, ymax, cls_id]:
                    line += "," + str(i)

                train_lst.write(line + '\n')



if __name__ == '__main__':
    gen_det_rec(classes, ratio)