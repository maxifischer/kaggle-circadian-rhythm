import time
import sys
import logging
import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def get_squares(image, size=16):
    squares = []
    i = 0
    while i < 512:
        j = 0
        while j < 512:
            squares.append([row[j:j+size] for row in image[i:i+size]])
            j += size
        i += size
    return squares


def write_squares_means(df, imdict, size=16):
    # paths = df['path']
    for square_num in range((512//size)**2):
        df['mean_sq{}'.format(square_num)] = np.nan

    i = 0
    
    start = time.time()
    for index in range(len(df)):
        sqrs = get_squares(imdict[df['path'][index]], size=size)
        for square_num in range((512//size)**2):
            df['mean_sq{}'.format(square_num)][index] = np.mean(sqrs[square_num])

        if i % 10 == 0:
            logging.info('Processed {} images in {} seconds'.
                         format(i, time.time()-start))
        i += 1

    logging.info('Done, total time {}s'.format(time.time() - start))


if __name__ == '__main__':
    datafile_npz_1 = sys.argv[1]
    datafile_npz_2 = sys.argv[2]
    datafile_csv = sys.argv[3]
    square_size = int(sys.argv[4])
    output = sys.argv[5]

    with np.load(datafile_npz_1) as im_data:
        image_dict = dict(zip(im_data['image_ids'], im_data['image_stack']))
    logging.info('Loaded {} images'.format(len(image_dict)))

    with np.load(datafile_npz_2) as im_data:
        image_dict2 = dict(zip(im_data['image_ids'], im_data['image_stack']))
        logging.info('Loaded {} images'.format(len(image_dict2)))

    image_dict.update(image_dict2)

    logging.info('Total images loaded: {}'.format(len(image_dict)))

    time_df = pd.read_csv(datafile_csv)
    time_df['path'] = time_df['Image.No.'].map(lambda x: "141110A3.%04d" % x)
    time_df['loaded'] = time_df['path'].map(lambda x: x in image_dict)
    valid_time_df = time_df.query('loaded')

    valid_time_df['mean_intensity'] = valid_time_df['path'].map(
        lambda x: np.mean(image_dict[x]))
    valid_time_df['std_intensity'] = valid_time_df['path'].map(
        lambda x: np.std(image_dict[x]))

    write_squares_means(valid_time_df, imdict=image_dict, size=square_size)
    valid_time_df.to_csv(output)
