

class PosDataSet:
    """
    用于训练锥体和椎间盘代表点的数据类
    """
    with open(os.path.join(os.path.dirname(__file__), 'json_files/pos2int.json'), 'r') as file:
        POS2INT = json.load(file, object_hook=OrderedDict)
        POS2INT = {k: i for i, k in enumerate(POS2INT)}

    def __init__(self, images: dict, metainfos: dict, annotation: dict, max_dist: float):
        """

        :param images: Dict[Tuple[str, str, str], np.ndarray]
        :param metainfos: Dict[Tuple[str, str, str], dict]
        :param annotation: Dict[Tuple[str, str, str], List[dict]]
        :param max_dist: 预测为正确的最大距离
        :param batch_size: batch的大小
        :param num_worker: DataLoader多进程的数量
        """
        self.images: List[Image] = []
        self.spacings = []
        self.coords = []

        non_hit_count = {}
        to_pil_image = transforms.ToPILImage()
        for k in annotation:
            image = to_pil_image(images[k])
            self.images.append(image)

            spacing = metainfos[k]['pixelSpacing']
            spacing = list(map(float, spacing.split('\\')))
            self.spacings.append(spacing)

            coord = torch.full([len(self.POS2INT), 2], -1, dtype=torch.long)
            for point in annotation[k]:
                identification = point['tag']['identification']
                if identification in self.POS2INT:
                    coord[self.POS2INT[identification]] = torch.tensor(point['coord'])
                elif identification in non_hit_count:
                    non_hit_count[identification] += 1
                else:
                    non_hit_count[identification] = 1
            self.coords.append(coord)
        if len(non_hit_count) > 0:
            print(non_hit_count)

        self.spacings: torch.Tensor = torch.tensor(self.spacings, dtype=torch.float32)
        self.coords: torch.Tensor = torch.stack(self.coords)
        self.max_dist = max_dist

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        gt_coord = self.coords[item]
        spacing = self.spacings[item]
        return image, gt_coord, spacing

    def gen_label(self, image: torch.Tensor, gt_coord: torch.Tensor, spacing: torch.Tensor):
        coord = torch.where(image.squeeze() < np.inf)
        coord = torch.stack(coord, dim=1).reshape(image.size(1), image.size(2), 2)
        dist = []
        for point in gt_coord:
            dist.append((((coord - point) * spacing) ** 2).sum(dim=-1).sqrt())
        dist = torch.stack(dist, dim=-1)
        return dist < self.max_dist

    @staticmethod
    def resize(size, image: Image, gt_coord, spacing):
        resize = transforms.Resize(size)
        image = resize(image)
        ratio = torch.tensor([size[0] / image.size[0], size[1] / image.size[1]])
        gt_coord = gt_coord * ratio
        spacing = spacing * ratio
        return image, gt_coord, spacing

    def collate_fn(self, data):
        images, gt_coords, spacings = [], [], []
        max_width, max_height = 0, 0
        for x, y, z in data:
            max_width = max(max_width, x.size[0])
            max_height = max(max_height, x.size[1])
            images.append(x)
            gt_coords.append(y)
            spacings.append(z)

        labels = []
        to_tensor = transforms.ToTensor()
        for i in range(len(images)):
            image, gt_coord, spacing = self.resize(images[i], gt_coords[i], spacings[i])
            image = to_tensor(image)

            labels.append(label)
            images[i] = image
        return (images,), (labels,)


class PosDataLoader(DataLoader):
    def __init__(self, images: dict, metainfos: dict, annotation: dict, max_dist: float, batch_size, num_workers=0):
        data_set = PosDataSet(images, metainfos, annotation, max_dist)
        super().__init__(data_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True,
                         collate_fn=data_set.collate_fn)