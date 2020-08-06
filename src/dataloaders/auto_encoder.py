import mlconfig
from torch.utils.data import DataLoader, Dataset
from torchface.utils.utils import find_ext
from torchvision.transforms import Compose, Resize, ToTensor
from .utils import get_train_valid_split_sampler


@mlconfig.register(name="AutoEncoderDataLoader")
class AutoEncoderDataLoader(DataLoader):
    '''
    Document here: https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
    '''

    def __init__(self, transformer=None, data_path=None, valide_ratio=0.1, *kwargs):
        dataset = AutoEncoderDataset(data_path)

        sampler = get_train_valid_split_sampler(dataset) if valide_ratio != 0 else None
        super().__init__(dataset, sampler=sampler)


class AutoEncoderDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_list = find_ext(data_path, ['.jpg', '.png'])
        self.transformer = AutoEncoderTransformer()

    def __getitem__(self, index):
        img = cv2.imread(self.data_list[index])
        img_torch = self.transformer

    def __len__(self):
        return len(self.data_list)


@mlconfig.register()
class AutoEncoderTransformer(Compose):
    '''
    Actually, Compose did nothing special, it just collect the input functions(here is Resize and ToTensor),
    and when you call it, it pass the data through all of the collected function orderly.
    Source code is here: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose
    '''
    def __init__(self, img_size):
        super().__init__([
            Resize(img_size, img_size),
            ToTensor()
        ])

@mlconfig.register()
class AutoEncoderTransformer2():
    '''
    It is th alternative of AutoEncoderTransformer, for that if you son't want to use Compose
    '''
    def __init__(self, img_size):
        self.to_tensor = ToTensor()
        self.resize = Resize(img_size, img_size)

    def __call__(self, image):
        img = self.resize(image)
        img = self.to_tensor(img)
        return img

    