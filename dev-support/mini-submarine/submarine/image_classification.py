# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
import os
import random
import tarfile
import time

import mxnet as mx
import numpy as np
from mxnet import autograd as ag
from mxnet import gluon, profiler
from mxnet.contrib.io import DataLoaderIter
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.model_zoo import vision as models
from mxnet.metric import Accuracy, CompositeEvalMetric, TopKAccuracy
from mxnet.test_utils import get_cifar10, get_mnist_iterator

# logging
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler("image-classification.log")
logger = logging.getLogger()
logger.addHandler(fh)
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logging.debug("\n%s", "-" * 100)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fh.setFormatter(formatter)

# CLI
parser = argparse.ArgumentParser(description="Train a model for image classification.")
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="dataset to use. options are mnist, cifar10, caltech101, imagenet and dummy.",
)
parser.add_argument(
    "--data-dir",
    type=str,
    default="",
    help="training directory of imagenet images, contains train/val subdirs.",
)
parser.add_argument(
    "--num-worker",
    "-j",
    dest="num_workers",
    default=4,
    type=int,
    help="number of workers for dataloader",
)
parser.add_argument(
    "--batch-size", type=int, default=32, help="training batch size per device (CPU/GPU)."
)
parser.add_argument(
    "--gpus",
    type=str,
    default="",
    help='ordinates of gpus to use, can be "0,1,2" or empty for cpu only.',
)
parser.add_argument("--epochs", type=int, default=120, help="number of training epochs.")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate. default is 0.1.")
parser.add_argument(
    "--momentum", type=float, default=0.9, help="momentum value for optimizer, default is 0.9."
)
parser.add_argument(
    "--wd", type=float, default=0.0001, help="weight decay rate. default is 0.0001."
)
parser.add_argument("--seed", type=int, default=123, help="random seed to use. Default=123.")
parser.add_argument(
    "--mode",
    type=str,
    help="mode in which to train the model. options are symbolic, imperative, hybrid",
)
parser.add_argument(
    "--model", type=str, required=True, help="type of model to use. see vision_model for options."
)
parser.add_argument(
    "--use_thumbnail", action="store_true", help="use thumbnail or not in resnet. default is false."
)
parser.add_argument(
    "--batch-norm",
    action="store_true",
    help="enable batch normalization or not in vgg. default is false.",
)
parser.add_argument(
    "--use-pretrained", action="store_true", help="enable using pretrained model from gluon."
)
parser.add_argument(
    "--prefix",
    default="",
    type=str,
    help="path to checkpoint prefix, default is current working dir",
)
parser.add_argument(
    "--start-epoch", default=0, type=int, help="starting epoch, 0 for fresh training, > 0 to resume"
)
parser.add_argument(
    "--resume", type=str, default="", help="path to saved weight where you want resume"
)
parser.add_argument("--lr-factor", default=0.1, type=float, help="learning rate decay ratio")
parser.add_argument(
    "--lr-steps", default="30,60,90", type=str, help="list of learning rate decay epochs as in str"
)
parser.add_argument(
    "--dtype", default="float32", type=str, help="data type, float32 or float16 if applicable"
)
parser.add_argument(
    "--save-frequency",
    default=10,
    type=int,
    help="epoch frequence to save model, best model will always be saved",
)
parser.add_argument(
    "--kvstore", type=str, default="device", help="kvstore to use for trainer/module."
)
parser.add_argument(
    "--log-interval", type=int, default=50, help="Number of batches to wait before logging."
)
parser.add_argument(
    "--profile",
    action="store_true",
    help=(
        "Option to turn on memory profiling for front-end, "
        "and prints out the memory usage by python function at the end."
    ),
)
parser.add_argument(
    "--builtin-profiler", type=int, default=0, help="Enable built-in profiler (0=off, 1=on)"
)
opt = parser.parse_args()

# global variables
logger.info("Starting new image-classification task:, %s", opt)
mx.random.seed(opt.seed)
model_name = opt.model
dataset_classes = {"mnist": 10, "cifar10": 10, "caltech101": 101, "imagenet": 1000, "dummy": 1000}
batch_size, dataset, classes = opt.batch_size, opt.dataset, dataset_classes[opt.dataset]
context = [mx.gpu(int(i)) for i in opt.gpus.split(",")] if opt.gpus.strip() else [mx.cpu()]
num_gpus = len(context)
batch_size *= max(1, num_gpus)
lr_steps = [int(x) for x in opt.lr_steps.split(",") if x.strip()]
metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])
kv = mx.kv.create(opt.kvstore)


def get_cifar10_iterator(batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    get_cifar10()

    train = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        resize=resize,
        data_shape=data_shape,
        batch_size=batch_size,
        rand_crop=True,
        rand_mirror=True,
        num_parts=num_parts,
        part_index=part_index,
    )

    val = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        resize=resize,
        rand_crop=False,
        rand_mirror=False,
        data_shape=data_shape,
        batch_size=batch_size,
        num_parts=num_parts,
        part_index=part_index,
    )

    return train, val


def get_imagenet_transforms(data_shape=224, dtype="float32"):
    def train_transform(image, label):
        image, _ = mx.image.random_size_crop(
            image, (data_shape, data_shape), 0.08, (3 / 4.0, 4 / 3.0)
        )
        image = mx.nd.image.random_flip_left_right(image)
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return mx.nd.cast(image, dtype), label

    def val_transform(image, label):
        image = mx.image.resize_short(image, data_shape + 32)
        image, _ = mx.image.center_crop(image, (data_shape, data_shape))
        image = mx.nd.image.to_tensor(image)
        image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return mx.nd.cast(image, dtype), label

    return train_transform, val_transform


def get_imagenet_iterator(root, batch_size, num_workers, data_shape=224, dtype="float32"):
    """Dataset loader with preprocessing."""
    train_dir = os.path.join(root, "train")
    train_transform, val_transform = get_imagenet_transforms(data_shape, dtype)
    logging.info("Loading image folder %s, this may take a bit long...", train_dir)
    train_dataset = ImageFolderDataset(train_dir, transform=train_transform)
    train_data = DataLoader(
        train_dataset, batch_size, shuffle=True, last_batch="discard", num_workers=num_workers
    )
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(os.path.expanduser(os.path.join(root, "val", "n01440764"))):
        user_warning = (
            "Make sure validation images are stored in one subdir per category, a helper script is"
            " available at https://git.io/vNQv1"
        )
        raise ValueError(user_warning)
    logging.info("Loading image folder %s, this may take a bit long...", val_dir)
    val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
    val_data = DataLoader(val_dataset, batch_size, last_batch="keep", num_workers=num_workers)
    return DataLoaderIter(train_data, dtype), DataLoaderIter(val_data, dtype)


def get_caltech101_data():
    url = "https://s3.us-east-2.amazonaws.com/mxnet-public/101_ObjectCategories.tar.gz"
    dataset_name = "101_ObjectCategories"
    data_folder = "data"
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    tar_path = mx.gluon.utils.download(url, path=data_folder)
    if not os.path.isdir(os.path.join(data_folder, "101_ObjectCategories")) or not os.path.isdir(
        os.path.join(data_folder, "101_ObjectCategories_test")
    ):
        tar = tarfile.open(tar_path, "r:gz")
        tar.extractall(data_folder)
        tar.close()
        print("Data extracted")
    training_path = os.path.join(data_folder, dataset_name)
    testing_path = os.path.join(data_folder, f"{dataset_name}_test")
    return training_path, testing_path


def get_caltech101_iterator(batch_size, num_workers, dtype):
    def transform(image, label):
        # resize the shorter edge to 224, the longer edge will be greater or equal to 224
        resized = mx.image.resize_short(image, 224)
        # center and crop an area of size (224,224)
        cropped, crop_info = mx.image.center_crop(resized, (224, 224))
        # transpose the channels to be (3,224,224)
        transposed = mx.nd.transpose(cropped, (2, 0, 1))
        return transposed, label

    training_path, testing_path = get_caltech101_data()
    dataset_train = ImageFolderDataset(root=training_path, transform=transform)
    dataset_test = ImageFolderDataset(root=testing_path, transform=transform)

    train_data = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=num_workers)
    test_data = DataLoader(dataset_test, batch_size, shuffle=False, num_workers=num_workers)
    return DataLoaderIter(train_data), DataLoaderIter(test_data)


class DummyIter(mx.io.DataIter):
    def __init__(self, batch_size, data_shape, batches=100):
        super().__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.label_shape = (batch_size,)
        self.provide_data = [("data", self.data_shape)]
        self.provide_label = [("softmax_label", self.label_shape)]
        self.batch = mx.io.DataBatch(
            data=[mx.nd.zeros(self.data_shape)], label=[mx.nd.zeros(self.label_shape)]
        )
        self._batches = 0
        self.batches = batches

    def next(self):
        if self._batches < self.batches:
            self._batches += 1
            return self.batch
        else:
            self._batches = 0
            raise StopIteration


def dummy_iterator(batch_size, data_shape):
    return DummyIter(batch_size, data_shape), DummyIter(batch_size, data_shape)


class ImagePairIter(mx.io.DataIter):
    def __init__(
        self, path, data_shape, label_shape, batch_size=64, flag=0, input_aug=None, target_aug=None
    ):
        def is_image_file(fn):
            return any(fn.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])

        super().__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.label_shape = (batch_size,) + label_shape
        self.input_aug = input_aug
        self.target_aug = target_aug
        self.provide_data = [("data", self.data_shape)]
        self.provide_label = [("label", self.label_shape)]
        self.filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]
        self.count = 0
        self.flag = flag
        random.shuffle(self.filenames)

    def next(self):
        from PIL import Image

        if self.count + self.batch_size <= len(self.filenames):
            data = []
            label = []
            for i in range(self.batch_size):
                fn = self.filenames[self.count]
                self.count += 1
                image = Image.open(fn).convert("YCbCr").split()[0]
                if image.size[0] > image.size[1]:
                    image = image.transpose(Image.TRANSPOSE)
                image = mx.nd.expand_dims(mx.nd.array(image), axis=2)
                target = image.copy()
                for aug in self.input_aug:
                    image = aug(image)
                for aug in self.target_aug:
                    target = aug(target)
                data.append(image)
                label.append(target)

            data = mx.nd.concat(*[mx.nd.expand_dims(d, axis=0) for d in data], dim=0)
            label = mx.nd.concat(*[mx.nd.expand_dims(d, axis=0) for d in label], dim=0)
            data = [mx.nd.transpose(data, axes=(0, 3, 1, 2)).astype("float32") / 255]
            label = [mx.nd.transpose(label, axes=(0, 3, 1, 2)).astype("float32") / 255]

            return mx.io.DataBatch(data=data, label=label)
        else:
            raise StopIteration

    def reset(self):
        self.count = 0
        random.shuffle(self.filenames)


def get_model(model, ctx, opt):
    """Model initialization."""
    kwargs = {"ctx": ctx, "pretrained": opt.use_pretrained, "classes": classes}
    if model.startswith("resnet"):
        kwargs["thumbnail"] = opt.use_thumbnail
    elif model.startswith("vgg"):
        kwargs["batch_norm"] = opt.batch_norm

    net = models.get_model(model, **kwargs)
    if opt.resume:
        net.load_parameters(opt.resume)
    elif not opt.use_pretrained:
        if model in ["alexnet"]:
            net.initialize(mx.init.Normal())
        else:
            net.initialize(mx.init.Xavier(magnitude=2))
    net.cast(opt.dtype)
    return net


net = get_model(opt.model, context, opt)


def get_data_iters(dataset, batch_size, opt):
    """get dataset iterators"""
    if dataset == "mnist":
        train_data, val_data = get_mnist_iterator(
            batch_size, (1, 28, 28), num_parts=kv.num_workers, part_index=kv.rank
        )
    elif dataset == "cifar10":
        train_data, val_data = get_cifar10_iterator(
            batch_size, (3, 32, 32), num_parts=kv.num_workers, part_index=kv.rank
        )
    elif dataset == "imagenet":
        shape_dim = 299 if model_name == "inceptionv3" else 224

        if not opt.data_dir:
            raise ValueError(
                "Dir containing raw images in train/val is required for imagenet."
                'Please specify "--data-dir"'
            )

        train_data, val_data = get_imagenet_iterator(
            opt.data_dir, batch_size, opt.num_workers, shape_dim, opt.dtype
        )
    elif dataset == "caltech101":
        train_data, val_data = get_caltech101_iterator(batch_size, opt.num_workers, opt.dtype)
    elif dataset == "dummy":
        shape_dim = 299 if model_name == "inceptionv3" else 224
        train_data, val_data = dummy_iterator(batch_size, (3, shape_dim, shape_dim))
    return train_data, val_data


def test(ctx, val_data):
    metric.reset()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(
            batch.data[0].astype(opt.dtype, copy=False), ctx_list=ctx, batch_axis=0
        )
        label = gluon.utils.split_and_load(
            batch.label[0].astype(opt.dtype, copy=False), ctx_list=ctx, batch_axis=0
        )
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()


def update_learning_rate(lr, trainer, epoch, ratio, steps):
    """Set the learning rate to the initial value decayed by ratio every N epochs."""
    new_lr = lr * (ratio ** int(np.sum(np.array(steps) < epoch)))
    trainer.set_learning_rate(new_lr)
    return trainer


def save_checkpoint(epoch, top1, best_acc):
    if opt.save_frequency and (epoch + 1) % opt.save_frequency == 0:
        fname = os.path.join(opt.prefix, "%s_%d_acc_%.4f.params" % (opt.model, epoch, top1))
        net.save_parameters(fname)
        logger.info("[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f", epoch, fname, top1)
    if top1 > best_acc[0]:
        best_acc[0] = top1
        fname = os.path.join(opt.prefix, "%s_best.params" % (opt.model))
        net.save_parameters(fname)
        logger.info("[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f", epoch, fname, top1)


def train(opt, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    train_data, val_data = get_data_iters(dataset, batch_size, opt)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(),
        "sgd",
        optimizer_params={
            "learning_rate": opt.lr,
            "wd": opt.wd,
            "momentum": opt.momentum,
            "multi_precision": True,
        },
        kvstore=kv,
    )
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    total_time = 0
    num_epochs = 0
    best_acc = [0]
    for epoch in range(opt.start_epoch, opt.epochs):
        trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(
                batch.data[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0
            )
            label = gluon.utils.split_and_load(
                batch.label[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0
            )
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                ag.backward(Ls)
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if opt.log_interval and not (i + 1) % opt.log_interval:
                name, acc = metric.get()
                logger.info(
                    "Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f, %s=%f"
                    % (
                        epoch,
                        i,
                        batch_size / (time.time() - btic),
                        name[0],
                        acc[0],
                        name[1],
                        acc[1],
                    )
                )
            btic = time.time()

        epoch_time = time.time() - tic

        # First epoch will usually be much slower than the subsequent epics,
        # so don't factor into the average
        if num_epochs > 0:
            total_time = total_time + epoch_time
        num_epochs = num_epochs + 1

        name, acc = metric.get()
        logger.info("[Epoch %d] training: %s=%f, %s=%f" % (epoch, name[0], acc[0], name[1], acc[1]))
        logger.info("[Epoch %d] time cost: %f" % (epoch, epoch_time))
        name, val_acc = test(ctx, val_data)
        logger.info(
            "[Epoch %d] validation: %s=%f, %s=%f"
            % (epoch, name[0], val_acc[0], name[1], val_acc[1])
        )

        # save model if meet requirements
        save_checkpoint(epoch, val_acc[0], best_acc)
    if num_epochs > 1:
        print(f"Average epoch time: {float(total_time) / (num_epochs - 1)}")


def main():
    if opt.builtin_profiler > 0:
        profiler.set_config(profile_all=True, aggregate_stats=True)
        profiler.set_state("run")
    if opt.mode == "symbolic":
        data = mx.sym.var("data")
        if opt.dtype == "float16":
            data = mx.sym.Cast(data=data, dtype=np.float16)
        out = net(data)
        if opt.dtype == "float16":
            out = mx.sym.Cast(data=out, dtype=np.float32)
        softmax = mx.sym.SoftmaxOutput(out, name="softmax")
        mod = mx.mod.Module(softmax, context=context)
        train_data, val_data = get_data_iters(dataset, batch_size, opt)
        mod.fit(
            train_data,
            eval_data=val_data,
            num_epoch=opt.epochs,
            kvstore=kv,
            batch_end_callback=mx.callback.Speedometer(batch_size, max(1, opt.log_interval)),
            epoch_end_callback=mx.callback.do_checkpoint("image-classifier-%s" % opt.model),
            optimizer="sgd",
            optimizer_params={
                "learning_rate": opt.lr,
                "wd": opt.wd,
                "momentum": opt.momentum,
                "multi_precision": True,
            },
            initializer=mx.init.Xavier(magnitude=2),
        )
        mod.save_parameters("image-classifier-%s-%d-final.params" % (opt.model, opt.epochs))
    else:
        if opt.mode == "hybrid":
            net.hybridize()
        train(opt, context)
    if opt.builtin_profiler > 0:
        profiler.set_state("stop")
        print(profiler.dumps())


if __name__ == "__main__":
    if opt.profile:
        import hotshot
        import hotshot.stats

        prof = hotshot.Profile(f"image-classifier-{opt.model}-{opt.mode}.prof")
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load(f"image-classifier-{opt.model}-{opt.mode}.prof")
        stats.strip_dirs()
        stats.sort_stats("cumtime", "calls")
        stats.print_stats()
    else:
        main()
