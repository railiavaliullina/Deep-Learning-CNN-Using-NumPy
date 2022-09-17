from configs.config import cfg
from utils.data_utils import get_data
from utils.visualization_utils import show_image


def check_image_by_id(idx, dl):
    batch_generator_ = dl.batch_generator()
    for batch in batch_generator_:
        image = batch[0][idx]
        show_image(image)


if __name__ == '__main__':
    print(f'Current dataset: {cfg["dataset"]}')

    train_dl = get_data(cfg, dataset_type='train')
    test_dl = get_data(cfg, dataset_type='test')

    # choose data_type to check
    data_type = 'test'  # ['train', 'test']

    if data_type == 'train':
        dl_to_check = train_dl
    else:
        dl_to_check = test_dl

    check_image_by_id(idx=0, dl=dl_to_check)
