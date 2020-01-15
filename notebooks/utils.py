from functools import partial
from pathlib import Path
from multiprocessing import Pool
import os
import shutil
import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile
import soundfile as sf
from tqdm import tqdm_notebook as tqdm
import torch.nn.functional as F
from fastai.basic_data import DatasetType
import matplotlib.pyplot as plt

def read_file(filename, path='', sample_rate=None, trim=False):
    ''' Reads in a wav file and returns it as an np.float32 array in the range [-1,1] '''
    filename = Path(path) / filename
    # file_sr, data = wavfile.read(filename)
    data,file_sr = sf.read(filename)
    # print(type(data))
    data=data.astype('float32')
    if data.dtype == np.int16:
        data = np.float32(data) / np.iinfo(np.int16).max
    elif data.dtype != np.float32:
        raise OSError('Encounted unexpected dtype: {}'.format(data.dtype))
    if sample_rate is not None and sample_rate != file_sr:
        if len(data) > 0:
            data = librosa.core.resample(data, file_sr, sample_rate, res_type='kaiser_fast')
        file_sr = sample_rate
    if trim and len(data) > 1:
        data = librosa.effects.trim(data, top_db=40)[0]
    return data, file_sr


def write_file(data, filename, path='', sample_rate=44100):
    ''' Writes a wav file to disk stored as int16 '''
    filename = Path(path) / filename
    if data.dtype == np.int16:
        int_data = data
    elif data.dtype == np.float32:
        int_data = np.int16(data * np.iinfo(np.int16).max)
    else:
        raise OSError('Input datatype {} not supported, use np.float32'.format(data.dtype))
    wavfile.write(filename, sample_rate, int_data)


def load_audio_files(path, filenames=None, sample_rate=None, trim=False):
    '''
    Loads in audio files and resamples if necessary.
    
    Args:
        path (str or PosixPath): directory where the audio files are located
        filenames (list of str): list of filenames to load. if not provided, load all 
                                 files in path
        sampling_rate (int): if provided, audio will be resampled to this rate
        trim (bool): 
    
    Returns:
        list of audio files as numpy arrays, dtype np.float32 between [-1, 1]
    '''
    path = Path(path)
    if filenames is None:
        filenames = sorted(list(f.name for f in path.iterdir()))
    files = []
    for filename in tqdm(filenames, unit='files'):
        data, file_sr = read_file(filename, path, sample_rate=sample_rate, trim=trim)
        files.append(data)
    return files
    
        
def _resample(filename, src_path, dst_path, sample_rate=16000, trim=True):
    data, sr = read_file(filename, path=src_path, sample_rate=sample_rate, trim=trim)
    write_file(data, filename, path=dst_path, sample_rate=sample_rate)
    

def resample_path(src_path, dst_path, **kwargs):
    transform_path(src_path, dst_path, _resample, **kwargs)    
    

def _to_mono(filename, dst_path):
    data, sr = read_file(filename)
    if len(data.shape) > 1:
        data = librosa.core.to_mono(data.T) # expects 2,n.. read_file returns n,2
    write_file(data, dst_path/filename.name, sample_rate=sr)


def convert_to_mono(src_path, dst_path, processes=None):
    src_path, dst_path = Path(src_path), Path(dst_path)
    os.makedirs(dst_path, exist_ok=True)
    filenames = list(src_path.iterdir())
    convert_fn = partial(_to_mono, dst_path=dst_path)
    with Pool(processes=processes) as pool:
        with tqdm(total=len(filenames), unit='files') as pbar:
            for _ in pool.imap_unordered(convert_fn, filenames):
                pbar.update()

def log_mel_spec_tfm(fname, src_path, dst_path):
    x, sample_rate = read_file(fname, src_path)
    
    n_fft = 1024
    hop_length = 256
    n_mels = 40
    fmin = 20
    fmax = sample_rate / 2 
    
    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft, 
                                                    hop_length=hop_length, 
                                                    n_mels=n_mels, power=2.0, 
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    dst_fname = dst_path / (fname[:-4] + '.png')
    plt.imsave(dst_fname, mel_spec_db)
                
                
def transform_path(src_path, dst_path, transform_fn, fnames=None, processes=None, delete=False, **kwargs):
    src_path, dst_path = Path(src_path), Path(dst_path)
    if dst_path.exists() and delete:
        shutil.rmtree(dst_path)
    os.makedirs(dst_path, exist_ok=True)
    
    _transformer = partial(transform_fn, src_path=src_path, dst_path=dst_path, **kwargs)
    if fnames is None:
        fnames = [f.name for f in src_path.iterdir()]
    with Pool(processes=processes) as pool:
        with tqdm(total=len(fnames), unit='files') as pbar:
            for _ in pool.imap_unordered(_transformer, fnames):
                pbar.update()


class RandomPitchShift():
    def __init__(self, sample_rate=22050, max_steps=3):
        self.sample_rate = sample_rate
        self.max_steps = max_steps
    def __call__(self, x):
        n_steps = np.random.uniform(-self.max_steps, self.max_steps)
        x = librosa.effects.pitch_shift(x, sr=self.sample_rate, n_steps=n_steps)
        return x


def _make_transforms(filename, src_path, dst_path, tfm_fn, sample_rate=22050, n_tfms=5):
    data, sr = read_file(filename, path=src_path)
    fn = Path(filename)
    # copy original file 
    new_fn = fn.stem + '_00.wav'
    write_file(data, new_fn, path=dst_path, sample_rate=sample_rate)
    # make n_tfms modified files
    for i in range(n_tfms):
        new_fn = fn.stem + '_{:02d}'.format(i+1) + '.wav'
        if not (dst_path/new_fn).exists():
            x = tfm_fn(data)
            write_file(x, new_fn, path=dst_path, sample_rate=sample_rate)


def pitch_shift_path(src_path, dst_path, max_steps, sample_rate, n_tfms=5):
    pitch_shifter = RandomPitchShift(sample_rate=sample_rate, max_steps=max_steps)
    transform_path(src_path, dst_path, _make_transforms, 
                   tfm_fn=pitch_shifter, sample_rate=sample_rate, n_tfms=n_tfms)
    
    
def rand_pad_crop(signal, pad_start_pct=0.1, crop_end_pct=0.5):
    r_pad, r_crop = np.random.rand(2)
    pad_start = int(pad_start_pct * r_pad * signal.shape[0])
    crop_end  = int(crop_end_pct * r_crop * signal.shape[0]) + 1
    return F.pad(signal[:-crop_end], (pad_start, 0), mode='constant')


def get_transforms(min_len=2048):
    def _train_tfm(x):
        x = rand_pad_crop(x)
        if x.shape[0] < min_len:
            x = F.pad(x, (0, min_len - x.shape[0]), mode='constant')
        return x
    
    def _valid_tfm(x):
        if x.shape[0] < min_len:
            x = F.pad(x, (0, min_len - x.shape[0]), mode='constant')
        return x
  
    return [_train_tfm],[_valid_tfm]


def save_submission(learn, filename, tta=False):
    fnames = [Path(f).name for f in learn.data.test_ds.x.items]
    get_predsfn = learn.TTA if tta else learn.get_preds
    preds = get_predsfn(ds_type=DatasetType.Test)[0]
    top_3 = np.array(learn.data.classes)[np.argsort(-preds, axis=1)[:, :3]]
    labels = [' '.join(list(x)) for x in top_3]
    df = pd.DataFrame({'fname': fnames, 'label': labels})
    df.to_csv(filename, index=False)
    return df


def precision(y_pred, y_true, thresh:float=0.2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between preds and targets"
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    return prec.mean()


def recall(y_pred, y_true, thresh:float=0.2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between preds and targets"
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    rec = TP/(y_true.sum(dim=1)+eps)
    return rec.mean()

	
import numpy as np
from keras.callbacks import Callback
from keras.models import clone_model
from tqdm import tqdm
from scipy.spatial.distance import cdist
import keras.backend as K


def get_bottleneck(classifier, samples):
    """Ripped from https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer"""
    inp = classifier.input  # input placeholder
    outputs = [layer.output for layer in classifier.layers]  # all layer outputs
    functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function

    # Get activations
    layer_outs = functor([samples, 0.])

    # Return bottleneck only
    return layer_outs[-2]


def preprocess_instances(downsampling, whitening=True):
    """This is the canonical preprocessing function for this project.
    1. Downsampling audio segments to desired sampling rate
    2. Whiten audio segments to 0 mean and fixed RMS (aka volume)
    """
    def preprocess_instances_(instances):
        instances = instances[:, ::downsampling, :]
        if whitening:
            instances = whiten(instances)
        return instances

    return preprocess_instances_


class BatchPreProcessor(object):
    """Wrapper class for instance and label pre-processing.
    This class implements a __call__ method that pre-process classifier-style batches (inputs, outputs) and siamese
    network-style batches ([input_1, input_2], outputs) identically.
    # Arguments
        mode: str. One of {siamese, classifier)
        instance_preprocessor: function. Pre-processing function to apply to input features of the batch.
        target_preprocessor: function. Pre-processing function to apply to output labels of the batch.
    """
    def __init__(self, mode, instance_preprocessor, target_preprocessor=lambda x: x):
        assert mode in ('siamese', 'classifier')
        self.mode = mode
        self.instance_preprocessor = instance_preprocessor
        self.target_preprocessor = target_preprocessor

    def __call__(self, batch):
        """Pre-processes a batch of samples."""
        if self.mode == 'siamese':
            ([input_1, input_2], labels) = batch

            input_1 = self.instance_preprocessor(input_1)
            input_2 = self.instance_preprocessor(input_2)

            labels = self.target_preprocessor(labels)

            return [input_1, input_2], labels
        elif self.mode == 'classifier':
            instances, labels = batch

            instances = self.instance_preprocessor(instances)

            labels = self.target_preprocessor(labels)

            return instances, labels
        else:
            raise ValueError


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(
        (1 - y_true) * K.square(y_pred) +
        y_true * K.square(K.maximum(margin - y_pred, 0))
    )


def whiten(batch, rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude i.e. volume."""
    if len(batch.shape) != 3:
        raise(ValueError, 'Input must be a 3D array of shape (n_segments, n_timesteps, 1).')

    # Subtract mean
    sample_wise_mean = batch.mean(axis=1)
    whitened_batch = batch - np.tile(sample_wise_mean, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    # Divide through
    sample_wise_rescaling = rms / np.sqrt(np.power(batch, 2).mean())
    whitened_batch = whitened_batch * np.tile(sample_wise_rescaling, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    return whitened_batch


def n_shot_task_evaluation(model, dataset, preprocessor, num_tasks, n, k, network_type='siamese', distance='euclidean'):
    """Evaluate a siamese network on k-way, n-shot classification tasks generated from a particular dataset.
    # Arguments
        model: Model to evaluate
        dataset: Dataset (currently LibriSpeechDataset only) from which to build evaluation tasks
        preprocessor: Preprocessing function to apply to samples
        num_tasks: Number of tasks to evaluate with
        n: Number of samples per class present in the support set
        k: Number of classes present in the support set
        network_type: Either 'siamese' or 'classifier'. This controls how to get the embedding function from the model
        distance: Either 'euclidean' or 'cosine'. This controls how to combine the support set samples for n > 1 shot
        tasks
    """
    # TODO: Faster/multiprocessing creation of tasks
    n_correct = 0

    if n == 1 and network_type == 'siamese':
        # Directly use siamese network to get pairwise verficiation score, minimum is closest
        for i_eval in tqdm(range(num_tasks)):
            query_sample, support_set_samples = dataset.build_n_shot_task(k, n)

            input_1 = np.stack([query_sample[0]] * k)[:, :, np.newaxis]
            input_2 = support_set_samples[0][:, :, np.newaxis]

            # Perform preprocessing
            # Pass an empty list to the labels parameter as preprocessor functions on batches not samples
            ([input_1, input_2], _) = preprocessor(([input_1, input_2], []))

            pred = model.predict([input_1, input_2])

            if np.argmin(pred[:, 0]) == 0:
                # 0 is the correct result as by the function definition
                n_correct += 1
    elif n > 1 or network_type == 'classifier':
        # Create encoder network from earlier layers
        if network_type == 'siamese':
           encoder = model.layers[2]
        elif network_type == 'classifier':
            encoder = clone_model(model)
            encoder.set_weights(model.get_weights())
            encoder.pop()
        else:
            raise(ValueError, 'mode must be one of (siamese, classifier)')

        for i_eval in tqdm((range(num_tasks))):
            query_sample, support_set_samples = dataset.build_n_shot_task(k, n)

            # Perform preprocessing
            query_instance = preprocessor.instance_preprocessor(query_sample[0].reshape(1, -1, 1))
            support_set_instances = preprocessor.instance_preprocessor(support_set_samples[0][:, :, np.newaxis])

            query_embedding = encoder.predict(query_instance)
            support_set_embeddings = encoder.predict(support_set_instances)

            if distance == 'euclidean':
                # Get mean position of support set embeddings
                # Assumes a label structure like [class_1]*n + [class_2]*n + ... + [class_k]*n
                # TODO: replace for loop with np.ufunc.reduceat
                mean_support_set_embeddings = []
                for i in range(0, n * k, n):
                    mean_support_set_embeddings.append(support_set_embeddings[i:i + n, :].mean(axis=0))
                mean_support_set_embeddings = np.stack(mean_support_set_embeddings)

                # Get euclidean distances between mean embeddings
                pred = np.sqrt(
                    np.power((np.concatenate([query_embedding] * k) - mean_support_set_embeddings), 2).sum(axis=1))
            elif distance == 'cosine':
                # Get "mean" position of support set embeddings. Do this by calculating the per-class mean angle
                # Assumes a label structure like [class_1]*n + [class_2]*n + ... + [class_k]*n
                magnitudes = np.linalg.norm(support_set_embeddings, axis=1, keepdims=True)
                unit_vectors = support_set_embeddings / magnitudes
                mean_support_set_unit_vectors = []
                for i in range(0, n * k, n):
                    mean_support_set_unit_vectors.append(unit_vectors[i:i + n, :].mean(axis=0))
                    # mean_support_set_magnitudes.append(magnitudes[i:i + n].sum() / n)

                mean_support_set_unit_vectors = np.stack(mean_support_set_unit_vectors)

                # Get cosine distance between angular-mean embeddings
                pred = cdist(query_embedding, mean_support_set_unit_vectors, 'cosine')
            elif distance == 'dot_product':
                # Get "mean" position of support set embeddings. Do this by calculating the per-class mean angle and
                # magnitude.
                # This is very similar to 'cosine' except in the case that two support set samples have the same angle,
                # in which case the one with the larger magnitude will be preffered
                # Assumes a label structure like [class_1]*n + [class_2]*n + ... + [class_k]*n
                magnitudes = np.linalg.norm(support_set_embeddings, axis=1, keepdims=True)
                unit_vectors = support_set_embeddings / magnitudes
                mean_support_set_unit_vectors = []
                mean_support_set_magnitudes = []
                for i in range(0, n * k, n):
                    mean_support_set_unit_vectors.append(unit_vectors[i:i + n, :].mean(axis=0))
                    mean_support_set_magnitudes.append(magnitudes[i:i + n].sum() / n)

                mean_support_set_unit_vectors = np.stack(mean_support_set_unit_vectors)
                mean_support_set_magnitudes = np.vstack(mean_support_set_magnitudes)
                mean_support_set_embeddings = mean_support_set_magnitudes * mean_support_set_unit_vectors

                # Get dot product between mean embeddings
                pred = np.dot(query_embedding[0, :][np.newaxis, :], mean_support_set_embeddings.T)
                # As dot product is a kind of similarity let's make this a "distance" by flipping the sign
                pred = -pred
            else:
                raise(ValueError, 'Distance must be in (euclidean, cosine, dot_product)')

            if np.argmin(pred) == 0:
                # 0 is the correct result as by the function definition
                n_correct += 1
    else:
        raise(ValueError, "n must be >= 1")

    return n_correct


class NShotEvaluationCallback(Callback):
    """Evaluate a siamese network on n-shot classification tasks after every epoch.
    # Arguments
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        dataset: LibriSpeechDataset. The dataset to generate the n-shot classification tasks from.
        preprocessor: function. The preprocessing function to apply to samples from the dataset.
        verbose: bool. Whether to enable verbose printing
        mode: str. One of {siamese, classifier}
    """
    def __init__(self, num_tasks, n_shot, k_way, dataset, preprocessor=lambda x: x, mode='siamese'):
        super(NShotEvaluationCallback, self).__init__()
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.dataset = dataset
        self.preprocessor = preprocessor

        assert mode in ('siamese', 'classifier')
        self.mode = mode
        # self.evaluator = evaluate_siamese_network if mode == 'siamese' else evaluate_classification_network

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        n_correct = n_shot_task_evaluation(self.model, self.dataset, self.preprocessor, self.num_tasks, self.n_shot,
                                           self.k_way, network_type=self.mode)

        n_shot_acc = n_correct * 1. / self.num_tasks
        logs['val_{}-shot_acc'.format(self.n_shot)] = n_shot_acc

        print('val_{}-shot_acc: {:.4f}'.format(self.n_shot, n_shot_acc))