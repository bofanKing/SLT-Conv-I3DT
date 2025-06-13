# coding: utf-8
"""
Data module
"""
import os
import sys
import random

import torch
from torchtext import data
from torchtext.data import Dataset, Iterator
import socket
from signjoey.dataset import SignTranslationDataset
from signjoey.vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)
import torch.nn.functional as F

def load_data(data_cfg: dict) -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    If you set ``random_dev_subset``, a random selection of this size is used
    from the dev development instead of the full development set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
        - gls_vocab: gloss vocabulary extracted from training data
        - txt_vocab: spoken text vocabulary extracted from training data
    """

    data_path = data_cfg.get("data_path", "./data")

    if isinstance(data_cfg["train"], list):
        train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
        dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
        test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
        pad_feature_size = sum(data_cfg["feature_size"])

    else:
        train_paths = os.path.join(data_path, data_cfg["train"])
        dev_paths = os.path.join(data_path, data_cfg["dev"])
        test_paths = os.path.join(data_path, data_cfg["test"])
        pad_feature_size = data_cfg["feature_size"]

    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    def tokenize_text(text):
        if level == "char":
            return list(text)
        else:
            return text.split()

    def tokenize_features(features):
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    # NOTE (Cihan): The something was necessary to match the function signature.
    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    sequence_field = data.RawField()
    signer_field = data.RawField()

    sgn_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        preprocessing=tokenize_features, # 将 [T, D] 切成 list of [D]
        tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        #pad_token={"cnn": torch.zeros((1024,)), "i3d": torch.zeros((1024,))},  # 二者一致
        pad_token=torch.zeros((pad_feature_size,)),
    )

    gls_field = data.Field(
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    txt_field = data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    train_data = SignTranslationDataset(
        path=train_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
        filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
        and len(vars(x)["txt"]) <= max_sent_length,
    )

# 特征向量替换
    my_feature_dir = './new_train' #I3D features

    feature_dict = {
        os.path.splitext(f)[0]: os.path.join(my_feature_dir, f)
        for f in os.listdir(my_feature_dir) if f.endswith('.pt')
    }

    new_examples = []
    replaced_count = 0
    removed_count = 0

    for example in train_data.examples:
        raw_seq_name = vars(example)["sequence"] # 原本视频对应的名字
        clean_name = raw_seq_name.replace("train/", "")  # 去掉前缀
        if clean_name in feature_dict: # 对于我提取到的视频信息
            feature_path = feature_dict[clean_name]

            cnn_feat_list = vars(example)["sgn"]
            cnn_feat = torch.stack(cnn_feat_list, dim=0)  # [T, 1024]
            i3d_feat = torch.load(feature_path, map_location="cpu") # I3D的特征向量
            if isinstance(i3d_feat, list):
                i3d_feat = torch.stack(i3d_feat, dim=0)

            T = cnn_feat.shape[0] # 对齐时间维度
            i3d_feat_aligned = F.interpolate(
                i3d_feat.unsqueeze(0).transpose(1, 2), size=T, mode='linear', align_corners=True
            ).squeeze(0).transpose(0, 1)

            fused_feat = 0.6 * cnn_feat + 0.4 * i3d_feat_aligned
            vars(example)["sgn"] = [f for f in fused_feat]  # 还原成 List[Tensor(1024)]

            new_examples.append(example)  # 保存当前的视频信息
            #print(f"✅ 成功替换特征: {clean_name}")
            replaced_count += 1
        else:
            #print(f"⚠️ 警告：特征文件 {raw_seq_name}.pt 未找到，保留原始特征。")
            removed_count += 1
    train_data.examples = new_examples # 没有被保存的信息就会被删除掉
    print("最后保留的train数据量为：",len(train_data)) # 7095
    #print(f"\n🎯 train总共成功替换了 {replaced_count} 个样本的特征。")
    #print(f"\n🎯 train总共成功删除了 {removed_count} 个样本的特征。")
    #input("train enter继续")

    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    gls_vocab_file = data_cfg.get("gls_vocab", None)
    txt_vocab_file = data_cfg.get("txt_vocab", None)

    gls_vocab = build_vocab(
        field="gls",
        min_freq=gls_min_freq,
        max_size=gls_max_size,
        dataset=train_data,
        vocab_file=gls_vocab_file,
    )
    txt_vocab = build_vocab(
        field="txt",
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        dataset=train_data,
        vocab_file=txt_vocab_file,
    )
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        train_data = keep

    dev_data = SignTranslationDataset(
        path=dev_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )

    my_feature_dir = './new_dev'
    feature_dict = {
        os.path.splitext(f)[0]: os.path.join(my_feature_dir, f)
        for f in os.listdir(my_feature_dir) if f.endswith('.pt')
    }

    new_examples = []
    replaced_count = 0
    remove_count = 0

    # 遍历 dev_data，尝试替换 sgn 特征
    for example in dev_data.examples:
        raw_seq_name = vars(example)["sequence"]
        clean_name = raw_seq_name.replace("dev/", "")  # 去掉前缀
        if clean_name in feature_dict:
            feature_path = feature_dict[clean_name]
            cnn_feat_list = vars(example)["sgn"]
            cnn_feat = torch.stack(cnn_feat_list, dim=0)  # [T, 1024]
            i3d_feat = torch.load(feature_path, map_location="cpu")  # I3D的特征向量
            if isinstance(i3d_feat, list):
                i3d_feat = torch.stack(i3d_feat, dim=0)

            T = cnn_feat.shape[0]  # 对齐时间维度
            i3d_feat_aligned = F.interpolate(
                i3d_feat.unsqueeze(0).transpose(1, 2), size=T, mode='linear', align_corners=True
            ).squeeze(0).transpose(0, 1)

            fused_feat = 0.6 * cnn_feat + 0.4 * i3d_feat_aligned
            vars(example)["sgn"] = [f for f in fused_feat]  # 还原成 List[Tensor(1024)]
            new_examples.append(example)
            #print(f"✅ 成功替换特征: {clean_name}")
            replaced_count += 1
        else:
            #print(f"⚠️ 警告：特征文件 {raw_seq_name}.pt 未找到，保留原始特征。")
            remove_count += 1

    dev_data.examples = new_examples
    print("最后保留的dev数据数量为：", len(dev_data)) # 519
    # 最终输出替换统计
    #print(f"\n🎯 dev总共成功替换了 {replaced_count} 个样本的特征。")
    #print(f"\n🎯 dev总共成功删除了 {remove_count} 个样本的特征。")
    #input("dev enter继续")

    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1:
        # select this many development examples randomly and discard the rest
        keep_ratio = random_dev_subset / len(dev_data)
        keep, _ = dev_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        dev_data = keep

    # check if target exists
    test_data = SignTranslationDataset(
        path=test_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )


    my_feature_dir = './new_test' # 如果是50个视频训练就是 './new_test_sample2' 全部就是'./new_test'
    # 创建特征文件映射字典
    feature_dict = {
        os.path.splitext(f)[0]: os.path.join(my_feature_dir, f)
        for f in os.listdir(my_feature_dir) if f.endswith('.pt')
    }


    new_examples = []
    replaced_count = 0  # 记录替换成功的样本数
    removed_count = 0

    #replaced_count = 0

    # 遍历 test_data，尝试替换 sgn 特征
    for example in test_data.examples:
        raw_seq_name = vars(example)["sequence"]
        clean_name = raw_seq_name.replace("test/", "")  # 去掉前缀
        if clean_name in feature_dict:
            feature_path = feature_dict[clean_name]
            cnn_feat_list = vars(example)["sgn"]
            cnn_feat = torch.stack(cnn_feat_list, dim=0)  # [T, 1024]
            i3d_feat = torch.load(feature_path, map_location="cpu")  # I3D的特征向量
            if isinstance(i3d_feat, list):
                i3d_feat = torch.stack(i3d_feat, dim=0)

            T = cnn_feat.shape[0]  # 对齐时间维度
            i3d_feat_aligned = F.interpolate(
                i3d_feat.unsqueeze(0).transpose(1, 2), size=T, mode='linear', align_corners=True
            ).squeeze(0).transpose(0, 1)

            fused_feat = 0.5 * cnn_feat + 0.5 * i3d_feat_aligned
            vars(example)["sgn"] = [f for f in fused_feat]  # 还原成 List[Tensor(1024)]
            new_examples.append(example)
            #print(f"✅ 成功替换特征: {clean_name}")
            replaced_count += 1
        else:
            #print(f"⚠️ 警告：特征文件 {raw_seq_name}.pt 未找到，删除原始特征。")
            removed_count += 1
    test_data.examples = new_examples
    print("最后保留的test数据数量为：",len(test_data)) #640
    # 最终输出替换统计
    #print(f"\n🎯 test总共成功替换了 {replaced_count} 个样本的特征。")
    #print(f"\n🎯 test总共成功删除了 {removed_count} 个样本的特征。")
    #input("test enter继续")
    gls_field.vocab = gls_vocab
    txt_field.vocab = txt_vocab
    return train_data, dev_data, test_data, gls_vocab, txt_vocab


# TODO (Cihan): I don't like this use of globals.
#  Need to find a more elegant solution for this it at some point.
# pylint: disable=global-at-module-level
global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)


def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: len(x.sgn),
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter


