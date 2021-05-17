import click # Command Line Interface 툴
from detection.src.steps import YOLOMAMLTraining

@click.command()
@click.option('--dataset_config', default='./detection/configs/coco.data')
@click.option('--model_config', default='./detection/configs/deep-tiny-yolo-3-way.cfg')
@click.option('--pretrained_weights', default='./data/weights/tiny.weights')
@click.option('--n_way', default=3)
@click.option('--n_shot', default=5)
@click.option('--n_query', default=10)
@click.option('--optimizer', default='Adam')
@click.option('--learning_rate', default=0.001)
@click.option('--approx', default=True)
@click.option('--task_update_num', default=2)
@click.option('--print_freq', default=100)
@click.option('--validation_freq', default=5000)
@click.option('--n_epoch', default=2)
@click.option('--n_episode', default=4)
@click.option('--objectness_threshold', default=0.5)
@click.option('--nms_threshold', default=0.3)
@click.option('--iou_threshold', default=0.5)
@click.option('--image_size', default=208)
@click.option('--random_seed', default=None)
@click.option('--output_dir', default='./output')

def main(
        dataset_config,
        model_config,
        pretrained_weights,
        n_way,
        n_shot,
        n_query,
        optimizer,
        learning_rate,
        approx,
        task_update_num,
        print_freq,
        validation_freq,
        n_epoch,
        n_episode,
        objectness_threshold,
        nms_threshold,
        iou_threshold,
        image_size,
        random_seed,
        output_dir,
):
    """
    Initializes the YOLOMAMLTraining step and executes it.
    Args:
        dataset_config (str): config 파일 경로
        model_config (str): 모델 정의 파일 경로
        pretrained_weights (str): pretrained weight 경로
        n_way (int):  support-set에서 샘플링할 class 갯수 
        n_shot (int): 한 class 당 샘플링할 데이터 수
        n_query (int): query 데이터 수
        optimizer (str): Adam, SGD등의 옵티마이져
        learning_rate (float): learning_rate
        approx (bool): 메타-역전파의 근사치를 사용할지 말지
        task_update_num (int): 각 에피소드에서업데이트할 갯수
        print_freq (int): 각 epoch에서 매 print_freq 에피소드마다 갱신된 상태를 프린트할 횟수
        validation_freq (int): 매 epoch마다, 검증셋에대해 모델을 평가할 빈도
        n_epoch (int): meta-training할 횟수
        n_episode (int): meta-training동안 각 epoch당 수행할 에피소드의 횟수
        objectness_threshold (float): 물체 예측시 물체를 탐지하는 threshold값
        nms_threshold (float): threshold for non maximum suppression, at evaluation time
        iou_threshold (float): threshold for intersection over union
        image_size (int): size of images (square)
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        output_dir (str): output경로
    """
    step = YOLOMAMLTraining(
        dataset_config,
        model_config,
        pretrained_weights,
        n_way,
        n_shot,
        n_query,
        optimizer,
        learning_rate,
        approx,
        task_update_num,
        print_freq,
        validation_freq,
        n_epoch,
        n_episode,
        objectness_threshold,
        nms_threshold,
        iou_threshold,
        image_size,
        random_seed,
        output_dir,
    )
    step.apply()

if __name__ == '__main__':
    main()
