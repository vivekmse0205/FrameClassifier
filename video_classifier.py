import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    def __init__(self, data, transform=None, label=None):
        """
        Args:
        data(numpy image)
        transform: Pytorch transforms for transformations and tensor conversion
        """
        self.data = data
        self.transform = transform
        self.label = label

    def __getitem__(self, index):
        """
        :param index: Index of the image
        :return: Transformed image
        """
        image = Image.fromarray(self.data)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return 1


class VideoFrameClassifier():
    def __init__(self, model_path):
        self.checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model = models.resnet34()
        self.model.fc = nn.Linear(512, 4)
        self.model.load_state_dict(self.checkpoint.get('state_dict'))
        self.model.eval()
        self.dataTransform = None
        self.frame_type_mapper = {
            '0': 'codewalk',
            '1': 'handwritten',
            '2': 'misc',
            '3': 'slides'
        }

    def transform_input(self, current_image):
        """Applies and transforms the given image to format for model
            @parameter
               current_contrast level (int/float)
                    current contrast level to be applied on image
                current_image:
                    Current image data
            @returns
                Transformed image to be feed into model
        """
        input_image = Image.fromarray(current_image)
        input_image = input_image.convert('RGB')
        input_image = input_image.resize(self.input_size, Image.ANTIALIAS)
        input_image = transforms.ToTensor()(input_image)
        width, height = input_image.size(1), input_image.size(2)
        input_image = input_image[0, :, :].view(1, width, height)
        input_image = input_image.unsqueeze(0)
        print(input_image.size())
        return input_image

    def get_prediction(self, image_data):
        """Get prediction of input image
            @parameter
               Image_data (numpy array)
                    current image for which prediction is needed
            @returns
                @type(numpy.int64)
                Class of current frame type
        """
        self.image = image_data
        self.model.eval()
        self.model = self.model.cpu()
        self.input_size = (224, 224)
        self.dataTransform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        self.transformedDataset = FrameDataset(data=image_data, transform=self.dataTransform)
        self.dataloader = torch.utils.data.DataLoader(
            self.transformedDataset,
            batch_size=1,
            shuffle=False
        )
        with torch.no_grad():
            for input_image in self.dataloader:
                prediction = self.model(input_image)
                prediction = np.argmax(prediction.detach().numpy())
        return prediction

    def classify_frame(self, image_data, ground_truth=None) -> str:
        """Classify the frame type(codewalk/misc/handwritten/others).
            @parameter
               image_data (numpy nd array)
                    Input Image
            @returns
                Type of frame - str
        """

        frame_type = self.get_prediction(image_data)
        frame_type = self.frame_type_mapper.get(str(frame_type))
        return frame_type
