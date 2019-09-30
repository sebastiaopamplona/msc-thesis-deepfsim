### WIKI_Age dataset 
- (from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- Parsed with MTCNN (https://github.com/ipazc/mtcnn) because some images were corrupted (original size: 62,328; size after MTCNN: 31,411)
- Available at:
https://drive.google.com/file/d/1o1ti_SSvYa-IQrBt_nYSS_i9SKzH7K85/view?usp=sharing

### Train the model on the WIKI_Age dataset 
1. Download and extract the WIKI_Age dataset
2. Clone the project
3. cd to the directory of the project
4. Run (python 3.6):

```bash
python train.py --dataset-dir="<PATH TO THE WIKI_AGE DATASET>"
```

### Test the DataGenerator
1. Run:

```bash
python tests/test_WIKI_DataGenerator.py
```
