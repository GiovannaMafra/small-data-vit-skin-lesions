import os, shutil

root = "tiny_imagenet/val"
src = os.path.join(root, "images")

if not os.path.isdir(src):
    print("Pasta 'images' não existe: o val provavelmente já foi reorganizado.")
else:
    with open(os.path.join(root, "val_annotations.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            fn, wnid = parts[0], parts[1]
            os.makedirs(os.path.join(root, wnid), exist_ok=True)
            shutil.move(os.path.join(src, fn), os.path.join(root, wnid, fn))
    os.rmdir(src)
    print("Pronto.")