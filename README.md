# VLM_interpretability
### Exploring how Vision-Language models interpret text in images.

## TODO:
- [ ] Generate datasets with random words (5 ~ 10 words)
  - Random `rotation`, `position`, `font_size`, `word` selected in each image
  - [X] Easy: `words = ["cat', "dog", "pet", "fox", "rat"]`
  - [ ] Medium_0: maybe 5 longer words?
  - [ ] Medium_1: 7 longer words?
  - [ ] Hard: 10 long words
- [ ] Train a linear classifier to distinguish which words are present in the image
  - Inputs are the generated images that are fed through the vision encoder first
- [ ] Extension: Test on images with 2 words on them instead of 1

## Generating New Datasets
### 1. Decompress template dataset
```bash
cd utils
# Run script to decompress template dataset
python decompress_datasets.py -i compressed_datasets/doggos_notext.zip
# Now the dataset is stored under /data
```

### 2. Create new dataset
```bash
python create_dataset.py -i data/doggos_notext/train -o data/<ENTER__NEW_NAME>
# Now the new dataset is stored /data
```
Note that `create_dataset.py` can take the following additional arguments in argparse:
  - `rand_position`: bool determining randomized position. Default is `True`
  - `rand_rotation`: bool determining randomized rotation. Default is `True`
  - `rand_font_size`: bool determining randomized font size. Default is `True`

## Adding New Datasets
If you have a small dataset that would be useful, follow the directions below to have it added into `compressed_datasets`
```bash
python compress_datasets.py -i <ENTER_PATH_TO_DATASET>
cd ..
# git lfs will be used to track large files. However, make sure the zipfile is under 500MB
git lfs track utils/compressed_datasets/<NAME_OF_DATASET_ZIPFILE>
# git commit and push
```
