import argparse
import jsonlines
from tqdm import tqdm
import json
from pycocotools.coco import COCO
import os

# this id_map is only for coco dataset which has 80 classes used for training but 90 categories in total.
# which change the start label -> 0
# {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}

def create_id_map(ori_map):
    id_map = {int(k)-1: i + 1 for i, (k, v) in enumerate(ori_map.items())}
    key_list = list(id_map.keys())
    val_list = list(id_map.values())
    return id_map, key_list, val_list

def coco_to_xyxy(bbox):
    x, y, width, height = bbox
    x1 = round(x, 2)
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]

def coco2odvg(args, id_map, key_list, val_list):
    coco = COCO(args.train_coco)
    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']: cat['name'] for cat in cats}
    metas = []

    for img_id, img_info in tqdm(coco.imgs.items()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            bbox = ann['bbox']
            bbox_xyxy = coco_to_xyxy(bbox)
            label = ann['category_id']
            category = nms[label]
            ind = val_list.index(label)
            label_trans = key_list[ind]
            instance_list.append({
                "bbox": bbox_xyxy,
                "label": label_trans,
                "category": category
            })
        metas.append(
            {
                "filename": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "detection": {
                "instances": instance_list
                }
            }
        )
    print("  == dump meta ...")
    train_odvg = args.train_coco.replace('.json','.jsonl')
    with jsonlines.open(train_odvg, mode="w") as writer:
        writer.write_all(metas)
    print("  == done.")

def create_dataset_json(root, train_odvg, train_label_map, train_mode,
                        val_coco, val_mode, output):
    train_root = os.path.join(root,'train')
    val_root = os.path.join(root,'val')
    label_map = {str(int(k)-1): v for k, v in json.loads(train_label_map).items()}
    print(label_map)
    dataset_info = {
        "train": [
            {
                "root": train_root,
                "anno": train_odvg,
                "label_map": label_map,
                "dataset_mode": train_mode
            }
        ],
        "val": [
            {
                "root": val_root,
                "anno": val_coco,
                "label_map": None,
                "dataset_mode": val_mode
            }
        ]
    }
    with open(output, "w") as f:
        json.dump(dataset_info, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("coco to odvg format.", add_help=True)
    parser.add_argument("--root", type=str, help="dataset root directory", 
                        default='DATASET/catdog_coco'
    )
    parser.add_argument("--train_coco", type=str, help="train set JSON file path",
                        default='DATASET/catdog_coco/annotations/instances_train.json'
    )
    
    parser.add_argument("--val_coco", type=str, help="validation set JSON file path", 
                        default='DATASET/catdog_coco/annotations/instances_val.json'
    )
    
    parser.add_argument("--ori_map", type=str, help="original label map in JSON format", 
                        default='{"1": "dog", "2": "cat"}'
    )
    
    parser.add_argument("--mode_train", type=str, help="training set dataset mode", 
                        default='odvg'
    )
    parser.add_argument("--mode_val", type=str, help="validation set dataset mode", 
                        default='coco'
    )
    
    parser.add_argument("--output", type=str, help="dataset path", 
                        default='config/data_dog_cat.json'
    )
    
    args = parser.parse_args()

    ori_map = json.loads(args.ori_map)

    id_map, key_list, val_list = create_id_map(ori_map)

    # 转换 COCO 数据集
    coco2odvg(args, id_map, key_list, val_list)

    # 创建 dataset.json 文件
    create_dataset_json(
        root=args.root,
        train_odvg=args.train_coco.replace('.json','.jsonl'),
        train_label_map=args.ori_map,
        train_mode=args.mode_train,
        val_coco=args.val_coco,
        val_mode=args.mode_val,
        output=args.output
    )
