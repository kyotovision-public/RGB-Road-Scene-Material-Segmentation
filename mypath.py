class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'kitti_advanced':
            return 'data/KITTI_Materials/'  # folder that contains KITTI_Materials dataset.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
