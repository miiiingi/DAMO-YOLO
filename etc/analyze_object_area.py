import os
import json
from collections import defaultdict

# ======================
# 설정
# ======================
JSON_DIR = r"C:\Users\User\workspace\Henkel\DAMO-YOLO\datasets\toy_sample_5_3_val_shrinked_original"  # json 폴더 경로
# 예: r"D:\workspace\HENKEL\labelme_jsons"

SMALL_TH = 32 * 32
MEDIUM_TH = 96 * 96


def bbox_area_from_points(points):
    (x1, y1), (x2, y2) = points
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return w, h, w * h


# 전체 통계
size_counter = defaultdict(int)
class_size_counter = defaultdict(lambda: defaultdict(int))
all_areas = []

for file in os.listdir(JSON_DIR):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(JSON_DIR, file)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue

        label = shape["label"]
        w, h, area = bbox_area_from_points(shape["points"])
        all_areas.append(area)

        if area < SMALL_TH:
            size = "small"
        elif area < MEDIUM_TH:
            size = "medium"
        else:
            size = "large"

        size_counter[size] += 1
        class_size_counter[label][size] += 1


# ======================
# 결과 출력
# ======================
print("\n📊 전체 객체 크기 분포")
print(f"\n📊 Folder Path: {JSON_DIR}")
total = sum(size_counter.values())
for k in ["small", "medium", "large"]:
    v = size_counter[k]
    print(f"{k:>6}: {v:4d}  ({v/total*100:.1f}%)")

print("\n📊 클래스별 크기 분포")
for cls, dist in class_size_counter.items():
    print(f"\n[{cls}]")
    for k in ["small", "medium", "large"]:
        print(f"  {k:>6}: {dist.get(k, 0)}")
