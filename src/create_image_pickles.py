import joblib
import glob
from tqdm import tqdm
import pyarrow.parquet as pq

if __name__ == '__main__':
    files = glob.glob('../input/bengaliai-cv19/train_*.parquet')
    for f in files:
        pf = pq.read_table(f)
        df = pf.to_pandas()
        image_ids = df.image_id.values
        df = df.drop('image_id', axis=1)
        image_array = df.values
        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[j, :], f'../input/image_pickles/{img_id}.pkl')