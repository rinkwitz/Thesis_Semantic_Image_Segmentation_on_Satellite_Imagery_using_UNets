import random
import torch
import tifffile
from Utils.Helpers import *
from pathlib import Path
import torchvision.transforms.functional as TF


class ImagesGeojsonData:
    def __init__(self, im_3band_orig_path, geojson_path):
        self.im_orig_path = im_3band_orig_path
        self.geojson_path = geojson_path
        self.im_npy_path = self.im_orig_path.parent / self.im_orig_path.name.replace('.tif', '.npy')
        self.mask_path = self.geojson_path.parent / self.geojson_path.name.replace('.geojson', '.npy')
        self.create_npy_data()
        self.bits = self.get_num_bits()
        assert self.bits is not None

    def create_npy_data(self):
        if not self.im_npy_path.exists():
            im = tifffile.imread(self.im_orig_path)
            np.save(self.im_npy_path, im)
        if not self.mask_path.exists():
            mask = get_segmentation_mask(self.geojson_path, self.im_orig_path)
            np.save(self.mask_path, mask)

    def get_num_bits(self):
        im = np.load(self.im_npy_path)
        if im.dtype == np.uint8:
            return 8
        elif im.dtype == np.uint16:
            return 16
        return None


class Dataloader:

    def __init__(self, dataset_name=None, im_type=None, path_dataset_folder=None, amount_train=.6, amount_val=0.2, amount_test=0.2,
                 delete_old_npy_data=False, data_fraction=1.0):
        self.dataset_name = dataset_name
        self.im_type = im_type
        self.path_dataset_folder = path_dataset_folder
        self.amount_train = amount_train
        self.amount_val = amount_val
        self.amount_test = amount_test
        assert self.amount_train + self.amount_val + self.amount_test == 1.0
        self.data_fraction = data_fraction
        self.mean, self.std = None, None

    def init(self):

        if self.dataset_name == 'SN1_Buildings':
            im_unformatted = '3band_AOI_1_RIO_img{}.tif'
            geojson_unformatted = 'Geo_AOI_1_RIO_img{}.geojson'

        num_entities = len([p for p in (self.path_dataset_folder / self.im_type).iterdir() if '.npy' not in p.name])
        assert num_entities == len(
            [p for p in (self.path_dataset_folder / 'geojson').iterdir() if '.npy' not in p.name])
        indices = random.sample(range(1, num_entities + 1), num_entities)
        self.amount_train *= self.data_fraction
        self.amount_val *= self.data_fraction
        self.amount_test *= self.data_fraction

        self.train_objs = [
            ImagesGeojsonData(Path(self.path_dataset_folder / self.im_type / im_unformatted.format(id)),
                                Path(self.path_dataset_folder / 'geojson' / geojson_unformatted.format(id))) for id in
            indices[:int(self.amount_train * num_entities)]]

        self.val_objs = [
            ImagesGeojsonData(Path(self.path_dataset_folder / self.im_type / im_unformatted.format(id)),
                                Path(self.path_dataset_folder / 'geojson' / geojson_unformatted.format(id))) for id in
            indices[int(self.amount_train * num_entities): int((self.amount_train + self.amount_val) * num_entities)]]

        self.test_objs = [
            ImagesGeojsonData(Path(self.path_dataset_folder / self.im_type / im_unformatted.format(id)),
                                Path(self.path_dataset_folder / 'geojson' / geojson_unformatted.format(id))) for id in
            indices[int((self.amount_train + self.amount_val) * num_entities):int(
                (self.amount_train + self.amount_val + self.amount_test) * num_entities)]]

    def save(self, path):
        save_pickle_obj(self, path)

    def load(self, path):
        dl_loaded = load_pickle_obj(path)
        self.dataset_name = dl_loaded.dataset_name
        self.im_type = dl_loaded.im_type
        self.path_dataset_folder = dl_loaded.path_dataset_folder
        self.amount_train = dl_loaded.amount_train
        self.amount_val = dl_loaded.amount_val
        self.amount_test = dl_loaded.amount_test
        self.train_objs = dl_loaded.train_objs
        self.val_objs = dl_loaded.val_objs
        self.test_objs = dl_loaded.test_objs

    def load_calculated_mean_and_std(self, params):
        if params['pretrained_on'] is None:
            mean_path = Path(f'tmp/mean_{self.dataset_name}.npy')
            std_path = Path(f'tmp/std_{self.dataset_name}.npy')
        else:
            mean_path = Path(f'tmp/mean_{params["pretrained_on"]}.npy')
            std_path = Path(f'tmp/std_{params["pretrained_on"]}.npy')
        if mean_path.exists() and self.mean is None:
            self.mean = torch.from_numpy(np.load(mean_path)).to(params['device'], torch.float32)
        if std_path.exists() and self.std is None:
            self.std = torch.from_numpy(np.load(std_path)).to(params['device'], torch.float32)

    def on_epoch_start(self, params):
        self.load_calculated_mean_and_std(params)
        self.train_indices = random.sample(range(len(self.train_objs)), len(self.train_objs))
        self.val_indices = [i for i in range(len(self.val_objs))]
        self.test_indices = [i for i in range(len(self.test_objs))]

    def get_mini_batch(self, num_batch, params, mode='train', weight=False, data_augmentation=False):

        if mode == 'train':
            obj = self.train_objs[num_batch]
            x = torch.from_numpy(np.load(obj.im_npy_path)).to(params['device'], torch.float32)
            y = torch.from_numpy(np.load(obj.mask_path)).to(params['device'], torch.uint8)
        elif mode == 'val':
            obj = self.val_objs[num_batch]
            x = torch.from_numpy(np.load(obj.im_npy_path)).to(params['device'], torch.float32)
            y = torch.from_numpy(np.load(obj.mask_path)).to(params['device'], torch.uint8)
        elif mode == 'test':
            obj = self.test_objs[num_batch]
            x = torch.from_numpy(np.load(obj.im_npy_path)).to(params['device'], torch.float32)
            y = torch.from_numpy(np.load(obj.mask_path)).to(params['device'], torch.uint8)
        h_x, w_x, ch_x = x.shape
        h_y, w_y = y.shape
        assert len(x.shape) == 3
        assert len(y.shape) == 2

        if data_augmentation:
            x, y = self.augment(x, y)
        x = self.standardize(x, obj)
        x = x.permute(2, 0, 1).view((1, ch_x, h_x, w_x))
        y = y.view((1, h_y, w_y))

        if weight:
            # current implementation works only with 2 categories
            assert params['num_categories'] == 2
            total = y.shape[1] * y.shape[2]
            cat_1 = torch.sum(y)
            return x, y, torch.tensor([total / (2 * (total - cat_1) + 1e-6), total / (2 * cat_1 + 1e-6)]).to(
                params['device'], torch.float32)
        else:
            return x, y

    def augment(self, x, y):

        angle = random.uniform(-180.0, 180.0)
        translate = [0, 0]
        scale = random.uniform(1.0, 1.015)
        shear = [random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)]
        flip_horizontal = random.random() < .5
        flip_vertical = random.random() < .5

        x = x.permute(2, 0, 1)
        x = TF.affine(x, angle=angle, translate=translate, scale=scale, shear=shear)
        if flip_horizontal:
            x = TF.hflip(x)
        if flip_vertical:
            x = TF.vflip(x)
        x = x.permute(1, 2, 0)

        h_y, w_y = y.shape
        y = y.view((1, h_y, w_y))
        y = TF.affine(y, angle=angle, translate=translate, scale=scale, shear=shear)
        if flip_horizontal:
            y = TF.hflip(y)
        if flip_vertical:
            y = TF.vflip(y)
        y = y.view((h_y, w_y))

        return x, y

    def standardize(self, x, obj):
        x = x / (2 ** obj.bits - 1)
        return torch.div(torch.sub(x, self.mean), self.std)
