import os
import h5py
import json
import torch
from torch.utils.data import Dataset
import torch.optim
import torch.utils.data

# data_folder = ''
# dataset_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  

# class COCOTrainDataset(Dataset):

#     def __init__(self, transform=None):

#         # Open hdf5 file where images are stored
#         self.h = h5py.File(os.path.join(data_folder, 'TRAIN_IMAGES_{}.hdf5'.format(dataset_name)), 'r')
#         self.imgs = self.h['images']

#         # Captions per image
#         self.cpi = self.h.attrs['captions_per_image']

#         # Load encoded captions (completely into memory)
#         with open(os.path.join('caption data','TRAIN_CAPTIONS_coco.json'), 'r') as j:
#             self.captions = json.load(j)

#         # Load caption lengths (completely into memory)
#         with open(os.path.join('caption data', 'TRAIN_CAPLENS_coco' + '.json'), 'r') as j:
#             self.caplens = json.load(j)
            
#         with open(os.path.join('caption data', 'TRAIN_names_coco' + '.json'), 'r') as j:
#             self.image_names = json.load(j)

#         # PyTorch transformation pipeline for the image (normalizing, etc.)
#         self.transform = transform

#         # Total number of datapoints
#         self.dataset_size = len(self.captions)

#     def __getitem__(self, i):
#         """
#         returns:
#         img: the image convereted into a tensor of shape (batch_size,3, 256, 256)
#         caption: the ground-truth caption of shape (batch_size, max_length)
#         caplen: the valid length (without padding) of the ground-truth caption of shape (batch_size,1)
#         """
#         # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
#         img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
#         img_name = self.image_names[i // self.cpi]
        
#         if self.transform is not None:
#             img = self.transform(img)

#         caption = torch.LongTensor(self.captions[i])
#         caplen = torch.LongTensor([self.caplens[i]])
        
#         return img, caption, caplen

#     def __len__(self):
#         return self.dataset_size


# class COCOValidationDataset(Dataset):

#     def __init__(self, transform=None):

#         # Open hdf5 file where images are stored
#         self.h = h5py.File(os.path.join('caption data', 'VAL_IMAGES_coco' + '.hdf5'), 'r')
#         self.imgs = self.h['images']

#         with open(os.path.join('caption data', 'VAL_names_coco' + '.json'), 'r') as j:
#             self.image_names = json.load(j)
            
#         with open('caption data/VAL_NAMES_IDS.json', 'r') as j:
#             self.name_ids = json.load(j)

#         # PyTorch transformation pipeline for the image (normalizing, etc.)
#         self.transform = transform

#         # Total number of datapoints
#         self.dataset_size = len(self.image_names)

#     def __getitem__(self, i):
#         """
#         returns:
#         img: the image convereted into a tensor of shape (batch_size,3, 256, 256)
#         image_id: the respective id for the image of shape (batch_size, 1)
#         """
#         img = torch.FloatTensor(self.imgs[i] / 255.)
#         img_name = self.image_names[i]
        
#         if self.transform is not None:
#             img = self.transform(img)
            
#         image_id = torch.LongTensor([self.name_ids[img_name]])
        
#         return img, image_id

#     def __len__(self):
#         return self.dataset_size


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, annotation_path, sentence_path, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # for flickr30k_entities sentences and bounding box annotations
        with open(os.path.join(data_folder, self.split + '_ANNOTATIONS_' + data_name + '.json'), 'r') as j:
            self.annotations = json.load(j)

        # Save word map to a JSON
        with open(os.path.join(data_folder, 'WORDMAP_' + data_name + '.json'), 'r') as j:
            self.word_map = json.load(j)

        # bounding box annotations
        with open(annotation_path, 'r') as j:
            self.annotation_data = json.load(j)

        # sentence phrases
        with open(sentence_path, 'r') as j:
            self.sentence_data = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # Maximum sentence length
        self.max_len = 50

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
      
        print(i // self.cpi, flush=True)
        file_key = self.annotations[i // self.cpi]
        print(file_key, flush=True)
        annotation = self.annotation_data[file_key]
        print("Annotations", annotation, flush=True)
        sentences = self.sentence_data[file_key]
        print("Sentneces", sentences, flush=True)
        print("single snentensdec", sentences[i // self.cpi], flush=True)
        sentence_map = sentences[i % self.cpi]
        sentence = sentence_map['sentence'].lower().split()
        
        # for sent in sentences:
        #     enc_sent = torch.LongTensor([self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in sent['sentence'].lower().split()] + [self.word_map['<end>']] + [self.word_map['<pad>']]     * (self.max_len - len(sent['sentence'].split())))
        #     print(enc_sent, sent['sentence'])

  
        print("caption_data---------", sentence_map['sentence'], flush=True)
        sentence_encoding = torch.LongTensor([self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in sentence] + [self.word_map['<end>']] + [self.word_map['<pad>']] * (self.max_len - len(sentence)))

        caption = torch.LongTensor(self.captions[i])

        print(sentence_encoding, caption)

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen #, sentence_map, annotation
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions #, sentence_map, annotation, sentences

    def __len__(self):
        return self.dataset_size
