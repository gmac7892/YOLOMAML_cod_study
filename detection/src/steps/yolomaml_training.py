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
'''파싱 또는 디폴트 값으로 초기화 시킨다'''
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
        
''' 여기서 새로운 변수 등장. device는 gpu를 주로 쓰기위한 스위치
writer는 SummaryWriter 클래스는 지정된 디렉토리에 이벤트 파일을 만들고 여기에 요약 및 이벤트를 추가할 수 있는 고급 API를 제공합니다. 
클래스는 파일 내용을 비동기적으로 업데이트합니다. 이를 통해 훈련 프로그램은 훈련 속도를 늦추지 않고 훈련 루프에서 직접 파일에 데이터를 추가하는 방법을 호출할 수 있다.'''

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=output_dir)

    def apply(self):
        """
        YOLOMAMLTraining step 실행
        Returns:
            더 높은 validation을 주는 모델의 전체상태를 제공하는 dict
        """
        set_and_print_random_seed(self.random_seed, True, self.checkpoint_dir) # 모델 재현을 위해 torch, numpy, cudnn 랜덤시드 고정

        data_config = parse_data_config(self.dataset_config) # config파일을 한줄씩 읽어와서, 객체형태로 리턴([gpus], [workers], [train], [val 경로], [클래스 수] 등...)
        train_path = data_config["train"] # data_cinfig에서 train경로 가져옴
        train_dict_path = data_config.get("train_dict_path", None) #dict은 아마도 파일과 라벨을 딕셔너리 형식으로 묶어 놓은것 같음
        valid_path = data_config.get("valid", None) # data_cinfig에서 val경로 가져옴
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
            path_to_data_file (str): 이미지 경로
            path_to_images_per_label (str): 라벨과 이미지가 묶인 딕셔너리를 피클파일의 경로
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
