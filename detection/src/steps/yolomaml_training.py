import os


import torch
from torch.utils.tensorboard.writer import SummaryWriter

from detection.src.loaders.data_manager import DetectionSetDataManager
from detection.src.yolo_maml import YOLOMAML
from utils import configs
from utils.io_utils import set_and_print_random_seed
from detection.src.yolov3.model import Darknet
from detection.src.yolov3.utils.parse_config import parse_data_config


class YOLOMAMLTraining():
    """
    This step handles the training of the algorithm on the base dataset
    """

    def __init__(
            self,
            dataset_config='yolov3/config/black.data',
            model_config='yolov3/config/yolov3.cfg',
            pretrained_weights=None,
            n_way=5,
            n_shot=5,
            n_query=16,
            optimizer='Adam',
            learning_rate=0.001,
            approx=True,
            task_update_num=3,
            print_freq=100,
            validation_freq=1000,
            n_epoch=100,
            n_episode=100,
            objectness_threshold=0.8,
            nms_threshold=0.4,
            iou_threshold=0.2,
            image_size=416,
            random_seed=None,
            output_dir=configs.save_dir,
    ):
        """
        Args:
            dataset_config (str): path to data config file
            model_config (str): path to model definition file
            pretrained_weights (str): path to a file containing pretrained weights for the model
            n_way (int): number of labels in a detection task
            n_shot (int): number of support data in each class in an episode
            n_query (int): number of query data in each class in an episode
            optimizer (str): must be a valid class of torch.optim (Adam, SGD, ...)
            learning_rate (float): learning rate fed to the optimizer
            approx (bool): whether to use an approximation of the meta-backpropagation
            task_update_num (int): number of updates inside each episode
            print_freq (int): inside an epoch, print status update every print_freq episodes
            validation_freq (int): inside an epoch, frequency with which we evaluate the model on the validation set
            n_epoch (int): number of meta-training epochs
            n_episode (int): number of episodes per epoch during meta-training
            objectness_threshold (float): at evaluation time, only keep boxes with objectness above this threshold
            nms_threshold (float): threshold for non maximum suppression, at evaluation time
            iou_threshold (float): threshold for intersection over union
            image_size (int): size of images (square)
            random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
            output_dir (str): path to experiments output directory
        """

        self.dataset_config = dataset_config
        self.model_config = model_config
        self.pretrained_weights = pretrained_weights
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.approx = approx
        self.task_update_num = task_update_num
        self.print_freq = print_freq
        self.validation_freq = validation_freq
        self.n_epoch = n_epoch
        self.n_episode = n_episode
        self.objectness_threshold = objectness_threshold
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.random_seed = random_seed
        self.checkpoint_dir = output_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=output_dir)

    def apply(self):
        """
        Execute the YOLOMAMLTraining step
        Returns:
            dict: a dictionary containing the whole state of the model that gave the higher validation accuracy
        """
        set_and_print_random_seed(self.random_seed, True, self.checkpoint_dir)

        data_config = parse_data_config(self.dataset_config)
        train_path = data_config["train"]
        train_dict_path = data_config.get("train_dict_path", None)
        valid_path = data_config.get("valid", None)
        valid_dict_path = data_config.get("valid_dict_path", None)

        base_loader = self._get_data_loader(train_path, train_dict_path)
        val_loader = self._get_data_loader(valid_path, valid_dict_path)

        model = self._get_model()

        return self._train(base_loader, val_loader, model)

    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def _train(self, base_loader, val_loader, model):
        """
        Trains the model on the base set
        Args:
            base_loader (torch.utils.data.DataLoader): data loader for base set
            val_loader (torch.utils.data.DataLoader): data loader for validation set
            model (YOLOMAML): neural network model to train
        Returns:
            dict: a dictionary containing the whole state of the model that gave the higher validation accuracy
        """
        optimizer = self._get_optimizer(model)

        for epoch in range(self.n_epoch):
            loss_dict = model.train_loop(base_loader, optimizer)

            self.plot_tensorboard(loss_dict, epoch)

            if epoch % self.print_freq == 0:
                print(
                    'Epoch {epoch}/{n_epochs} | Loss {loss}'.format(
                        epoch=epoch,
                        n_epochs=self.n_epoch,
                        loss=loss_dict['query_total_loss'],
                    )
                )

            if epoch % self.validation_freq == self.validation_freq - 1:
                precision, recall, average_precision, f1, ap_class = model.eval_loop(val_loader)

                self.writer.add_scalar('precision', precision.mean(), epoch)
                self.writer.add_scalar('recall', recall.mean(), epoch)
                self.writer.add_scalar('mAP', average_precision.mean(), epoch)
                self.writer.add_scalar('F1', f1.mean(), epoch)

        self.writer.close()

        model.base_model.save_darknet_weights(os.path.join(self.checkpoint_dir, 'final.weights'))

        return {'epoch': self.n_epoch, 'state': model.state_dict()}

    def _get_optimizer(self, model):
        """
        Get the optimizer from string self.optimizer
        Args:
            model (torch.nn.Module): the model to be trained
        Returns: a torch.optim.Optimizer object parameterized with model parameters
        """
        assert hasattr(torch.optim, self.optimizer), "The optimization method is not a torch.optim object"
        optimizer = getattr(torch.optim, self.optimizer)(model.parameters(), lr=self.learning_rate)

        return optimizer

    def _get_data_loader(self, path_to_data_file, path_to_images_per_label):
        """
        Args:
            path_to_data_file (str): path to file containing paths to images
            path_to_images_per_label (str): path to pickle file containing the dictionary of images per label
        Returns:
            torch.utils.data.DataLoader: samples data in the shape of a detection task
        """
        data_manager = DetectionSetDataManager(self.n_way, self.n_shot, self.n_query, self.n_episode, self.image_size)

        return data_manager.get_data_loader(path_to_data_file, path_to_images_per_label)

    def _get_model(self):
        """
        Returns:
            YOLOMAML: meta-model
        """

        base_model = Darknet(self.model_config, self.image_size, self.pretrained_weights)

        model = YOLOMAML(
            base_model,
            self.n_way,
            self.n_shot,
            self.n_query,
            self.image_size,
            approx=self.approx,
            task_update_num=self.task_update_num,
            train_lr=self.learning_rate,
            objectness_threshold=self.objectness_threshold,
            nms_threshold=self.nms_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
        )

        return model

    def plot_tensorboard(self, loss_dict, epoch):
        """
        Writes into summary the values present in loss_dict
        Args:
            loss_dict (dict): contains the different parts of the average loss on one epoch. Each key describes
            a part of the loss (ex: query_classification_loss) and each value is a 0-dim tensor. This dictionary is
            required to contain the keys 'support_total_loss' and 'query_total_loss' which contains respectively the
            total loss on the support set, and the total meta-loss on the query set
            epoch (int): global step value in the summary
        Returns:
        """
        for key, value in loss_dict.items():
            self.writer.add_scalar(key, value, epoch)

        return
