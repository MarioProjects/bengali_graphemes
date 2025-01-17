from shutil import copyfile
import fastai
from fastai.vision import *
from fastai.callbacks import *

#@dataclass
class CSVLogger(LearnerCallback):
    def __init__(self, learn, summary_writer, filename= 'history'):
        self.learn = learn
        self.path = self.learn.path/f'{filename}.csv'
        self.file = None
        self.summary_writer = summary_writer
        self.filename = filename

    @property
    def header(self):
        return self.learn.recorder.names

    def read_logged_file(self):
        return pd.read_csv(self.path)

    def on_train_begin(self, metrics_names: StrList, **kwargs: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        e = self.path.exists()
        self.file = self.path.open('a')
        if not e: self.file.write(','.join(self.header) + '\n')

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        self.write_stats([epoch, smooth_loss] + last_metrics)
        self.summary_writer.add_scalar('MetricTot/train', last_metrics[-1].item(), epoch)

    def on_train_end(self, **kwargs: Any) -> None:
        self.file.flush()
        self.file.close()
        copyfile(self.path, '{}.csv'.format(self.filename))

    def write_stats(self, stats: TensorOrNumList) -> None:
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.header, stats)]
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')