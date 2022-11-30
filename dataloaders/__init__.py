from dataloaders.datasets import kitti_advanced
from torch.utils.data import DataLoader
import torch

def make_data_loader(args, **kwargs):
    if args.dataset == 'kitti_advanced':
        if args.positional_encoding:
            def my_collate_fn(data):
                image_cat_list = []
                label_list = []
                mask_list = []
                for d in data:
                    image, images_remapped, label, u_map, v_map, mask = d.values()
                    images_remapped.append(image)
                    u_map = u_map[None,:,:]
                    v_map = v_map[None,:,:]
                    images_remapped.append(u_map)
                    images_remapped.append(v_map)
                    image_cat = torch.cat(images_remapped,dim=0)
                    image_cat_list.append(image_cat)
                    label_list.append(label)
                    mask_list.append(mask)
                image_cat = torch.stack(image_cat_list, dim=0)
                label = torch.stack(label_list, dim=0)
                mask = torch.stack(mask_list, dim=0)
                return {'image_cat':image_cat, 'label': label, 'mask': mask}
        else:
            def my_collate_fn(data):
                image_cat_list = []
                label_list = []
                mask_list = []
                for d in data:
                    image, images_remapped, label, u_map, v_map, mask = d.values()
                    images_remapped.append(image)
                    image_cat = torch.cat(images_remapped,dim=0)
                    image_cat_list.append(image_cat)
                    label_list.append(label)
                    mask_list.append(mask)
                image_cat = torch.stack(image_cat_list, dim=0)
                label = torch.stack(label_list, dim=0)
                mask = torch.stack(mask_list, dim=0)
                return {'image_cat':image_cat, 'label': label, 'mask':mask}

        train_set = kitti_advanced.KITTIAdvSegmentation(args, split='train')
        val_set = kitti_advanced.KITTIAdvSegmentation(args, split='val')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size_val, shuffle=False, collate_fn=my_collate_fn, drop_last=True, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class


    else:
        raise NotImplementedError

