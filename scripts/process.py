import glob
import os
import shutil
import xml.etree.ElementTree as ET
import csv


def fix_file_path(path):
    return path.replace(' ', '_')

DATASET_ROOT = '/home/vinit/data/human_seat_data_JAN2020_cleaned'
TRAINING_ROOT = os.path.join(DATASET_ROOT, 'TRAINING')
VALIDATION_ROOT = os.path.join(DATASET_ROOT, 'VALIDATION')

ANNOTATIONS_PATH = ['*', 'annotations', '*.xml']
IMAGES_PATH = ['*', 'images', '*.jpg']

TRAINING_ANNOTATIONS_PATH = os.path.join(TRAINING_ROOT, *ANNOTATIONS_PATH)
TRAINING_IMAGES_PATH = os.path.join(TRAINING_ROOT, *IMAGES_PATH)

VALIDATION_ANNOTATIONS_PATH = os.path.join(VALIDATION_ROOT, *ANNOTATIONS_PATH)
VALIDATION_IMAGES_PATH = os.path.join(VALIDATION_ROOT, *IMAGES_PATH)

training_annotations = glob.glob(TRAINING_ANNOTATIONS_PATH, recursive=True)
training_images = glob.glob(TRAINING_IMAGES_PATH, recursive=True)

validation_annotations = glob.glob(VALIDATION_ANNOTATIONS_PATH, recursive=True)
validation_images = glob.glob(VALIDATION_IMAGES_PATH, recursive=True)

TARGET_ROOT = os.path.join(DATASET_ROOT, 'out')

if not os.path.exists(TARGET_ROOT):
    os.makedirs(TARGET_ROOT)


def match_annotations_to_images(annotations, images_path):
    matching = {}

    for annotation in annotations:
        folder = os.path.dirname(annotation)
        file_name = os.path.basename(annotation)
        image_name = file_name.replace('.xml', '.jpg')
        image_path = os.path.join(folder, '..', 'images', image_name)
        if os.path.exists(image_path):
            matching[annotation] = image_path

    return matching


def remap_class(klass):
    PERSON = 'person'
    SEATBELT = 'seatbelt'
    MAPPING = {
        'person': PERSON,
        'Parson': PERSON,
        'front': PERSON,
        'parson': PERSON,
        'seatbelt': SEATBELT,
        'Seatbelt': SEATBELT
    }
    return MAPPING.get(klass)


def extract_objects_from_annotation(root, path):
    objects = []

    for o in root.findall('object'):
        klass = remap_class(o.find('name').text)

        if not klass:
            continue

        xmin = o.find('bndbox').find('xmin').text
        ymin = o.find('bndbox').find('ymin').text
        xmax = o.find('bndbox').find('xmax').text
        ymax = o.find('bndbox').find('ymax').text

        objects.append((path, xmin, ymin, xmax, ymax, klass))

    return objects


train_objects = []

for annotation, image in match_annotations_to_images(
        training_annotations, TRAINING_IMAGES_PATH).items():
    tree = ET.parse(annotation)
    root = tree.getroot()
    train_objects += extract_objects_from_annotation(root, image)


klasses = set()
for annotation in train_objects:
    klasses.add(annotation[5])


with open(os.path.join(TARGET_ROOT, 'annotations.csv'), 'w+') as f:
    writer = csv.writer(f)
    for annotation in train_objects:
        writer.writerow(annotation)

val_objects = []

for annotation, image in match_annotations_to_images(
        validation_annotations, VALIDATION_IMAGES_PATH).items():
    tree = ET.parse(annotation)
    root = tree.getroot()
    val_objects += extract_objects_from_annotation(root, image)


klasses = set()
for annotation in val_objects:
    klasses.add(annotation[5])


with open(os.path.join(TARGET_ROOT, 'val_annotations.csv'), 'w+') as f:
    writer = csv.writer(f)
    for annotation in val_objects:
        writer.writerow(annotation)

with open(os.path.join(TARGET_ROOT, 'classes.csv'), 'w+') as f:
    for k, klass in enumerate(list(klasses)):
        f.write(f'{klass},{k}\n')

# print(all_objects)
