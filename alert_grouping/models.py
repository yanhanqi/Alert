from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any, Literal
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from .preprocessing import BaseSequenceCollate, Vocabulary

class BaseParams:
    """This class is a base class for defining parameters for training a masked language model.
    Implementations of subclasses should in ther __init__ methods pass locals() to super().__init__() to
    save the parameters passed to them in the dictionary self.dict.

    Args:
        - kwargs (dict): A dictionary of keyword arguments representing the parameters.
    """

    def __init__(self, kwargs: dict) -> None:
        self.dict = {
            k: v for k, v in kwargs.items() if (k != "self") and not k.startswith("_")
        }

    def __setitem__(self, key: str, value: object) -> None:
        self.dict[key] = value

    def __getitem__(self, key: str) -> object:
        return self.dict[key]

    def __delitem__(self, key: str) -> None:
        del self.dict[key]

    def __repr__(self) -> str:
        return self.dict.__repr__()


class MaskedLangModelParams(BaseParams):
    """This class defines the parameters for training a masked language model on the AIT Alert dataset.
    Additionally to the arguments defined in the __init__ method, the following parameters are set:
    - d_model: The dimension of the model, calculated as n_heads * dim_per_head.
    - dim_feedforward: The dimension of the feedforward layer, calculated as n_heads * dim_per_head * feedforward_factor.
    - encoding_freqs: The frequencies used for the rotary encoding, calculated based on the encoding type and parameters.
    - id: The identifier for the model, generated based on the number of layers, number of heads, dimension per head, augmentation configuration and id_suffix.

    Note: The save_intervals are not added to the parameters dictionary but as an attribute to the object. Instead, in the parameters
        saved at the end of training the updates key is set to the number of updates at which the best model was saved.

    Args:
        - id_suffix (str): The identifier for the model.
        ### data params
        - augment (Literal | None): The configuration of AIT-ADS-A to use. If None, the original AIT Alert dataset is used. Default is "original".
        - context_size (int): The size of the context window. Default is 4096.
        - batch_size (int): The batch size for training. Default is 16.
        - features (tuple[Literal["short", "host"]]): The list of features to include in the model. Default is ("short", "host").
        - targets (tuple[Literal["short", "host"]]): The list of target variables to predict. Default is ("short", "host").
        - sampling (Literal["index", "time"]): The sampling method to use for creating batches. The option "time" is deprecated. Default is "index".
        - min_freq (int): The minimum frequency of a token to be included in the vocabulary. Default is 10.
        ### model params
        - n_heads (int): The number of attention heads in each layer of the model. Default is 4.
        - dim_per_head (int): The dimension of each attention head in the model. Default is 16.
        - num_layers (int): The number of layers in the model. Default is 1.
        - feedforward_factor (int): The factor by which to multiply the dimension of the model to get the dimension
            of the feedforward layer in the transformer encoder. Default is 4.
        - activation (Literal["relu", "gelu"]): The activation function to use in the feedforward layer. Default is "gelu".
        - gated_activation (bool): Whether to use a gated activation function in the feedforward layer. Default is True.
        - encoding (Literal["position", "raw_time"]): The encoding method to use. Default is "raw_time".
        - encoding_type (Literal["learned", "rotary"]): The type of encoding to use. The option "learned" is only available for the "position" encoding.
            Sinusoidal encoding is currently not implemented. Default is "rotary".
        - rotary_max_exp (int): The maximum exponent of the frequencies to use for the rotary encoding. For positional encoding this should be log2(context_size),
            for time encoding it should be log2 of the maximal reasonable timespan in a context window, e.g. 14 for overnight context windows. Default is 14.
            Ignored if encoding_type is "learned".
        - rotary_cutoff (float): The cutoff ratio for the frequencies to use for the rotary encoding. Default is 0.75. Ignored if encoding_type is "learned".
        - biases (bool): Whether to include biases in the model. Default is False.
        - head_bias (bool): Whether to use biases in the prediction head. Default is True.
        - tie_weights (bool): Whether to tie the weights of the input and output embeddings. Default is True.
        - emb_init_std (float): The standard deviation of the normal distribution to use for initializing the embeddings.
            If None, the standard deviation is set to 1/sqrt(d_model). Default is None.
        ### training params
        - save_intervals (list[tuple[int, int]]): A list of tuples specifying the number of model updates at which to start and end each save interval.
        - optimizer (Literal["sgd", "adam"]): The optimizer to use for training. Default is "adam".
        - scheduler (Literal["schedulefree", "linear"]): The scheduler to use for training. Default is "linear".
        - lr (float): The learning rate for the optimizer. Default is 5e-3.
        - warm_up_steps (int | float): The number of warm-up steps for the learning rate scheduler. Default is 200.
        - decay (float): The weight decay for the optimizer. Default is 0.1.
        - momentum (float): The momentum for the optimizer. Default is 0.9.
        - gamma (float | None): The gamma parameter for the focal loss function, if used. If None, the cross-entropy loss function is used.
            Focal loss is currently deprecated. Default is None.
        - class_balance (float): Inverse softmax temperature to be applied to class frequencies to obtain class balancing weights for the loss function.
            If 0, no class balancing is applied, positive values emphasize underrepresented classes, and negative values emphasize overrepresented classes. Default is 2.0.
        - target_ratio (float): The ratio of target tokens to mask in the input sequence. Default is 0.2.
        - mask_ratio (float): The ratio of target tokens to replace with the mask token in the input sequence. Default is 0.8.
        - perturb_ratio (float): The ratio of target tokens to replace with a random token in the input sequence. Default is 0.1.
        ### file params
        - path (str): The path to save the model checkpoints. Default is "saved_models".
        - log (str): The name of the log file to use. If None, logging is done to stdout. Default is "train".

    """

    def __init__(
        self,
        id_suffix: str,
        # data
        augment: str | None = "original",
        context_size: int = 4096,
        batch_size: int = 16,
        features: tuple[str] = ("short", "host"),
        targets: tuple[str] = ("short", "host"),
        sampling: Literal["index", "time"] = "index",
        min_freq: int = 10,
        # model
        n_heads: int = 4,
        dim_per_head: int = 16,
        num_layers: int = 1,
        feedforward_factor: int = 4,
        activation: Literal["relu", "gelu"] = "gelu",
        gated_activation: bool = True,
        encoding: Literal["position", "raw_time"] = "raw_time",
        encoding_type: Literal["learned", "rotary"] = "rotary",
        rotary_max_exp: int | None = 14,
        rotary_cutoff: float | None = 0.75,
        biases: bool = False,
        head_bias: bool = True,
        tie_weights: bool = True,
        emb_init_std: float = None,
        # training
        save_intervals: tuple[tuple[int, int]] = (
            (18000, 20000),
            (38000, 40000),
            (58000, 60000),
        ),
        optimizer: Literal["sgd", "adam"] = "adam",
        scheduler: Literal["schedulefree", "linear"] = "linear",
        lr: float = 5e-3,
        warm_up_steps: int = 200,
        decay: float = 0.1,
        momentum: float = 0.9,
        gamma: float | None = None,
        class_balance: float = 2.0,
        target_ratio: float = 0.2,
        mask_ratio: float = 0.8,
        perturb_ratio: float = 0.1,
        # files
        path: str = "saved_models",
        log: str = "train",
    ) -> None:
        super().__init__(locals())

        self["d_model"] = n_heads * dim_per_head
        self["dim_feedforward"] = n_heads * dim_per_head * feedforward_factor

        if encoding_type == "rotary":
            self["encoding_freqs"] = [
                2 ** (-i * 2.0 / dim_per_head * rotary_max_exp)
                for i in range(int(dim_per_head // 2 * rotary_cutoff))
            ]
        elif encoding_type == "learned":
            self["encoding_freqs"] = None
            self["rotary_max_exp"] = None
            self["rotary_cutoff"] = None

        if self["emb_init_std"] is None:
            self["emb_init_std"] = (n_heads * dim_per_head) ** -0.5

        self.save_intervals = save_intervals
        del self["save_intervals"]
        self["updates"] = save_intervals[-1][1]

        self["id"] = f"{num_layers}l_{n_heads}h_{dim_per_head}d_{augment}_{id_suffix}"
        del self["id_suffix"]


# model components


class MaskedLangModelEmbeddingLayer(nn.Module):
    """A module that represents the embedding layer for a masked language model.

    Args:
        features (list[str]): List of feature to use.
        encoding (str | None): Feature to use for positional encoding.
        vocabs (dict[str, Vocabulary]): Dictionary of feature vocabularies.
        dim (int): Dimension of the embeddings.
        init_std (float, optional): Standard deviation of the normal distribution used
            for initializing the embeddings. Defaults to 1.0.
        context_size (int, optional): Size of the context for positional encoding.
            Defaults to None.

    Attributes:
        dim (int): Dimension of the embeddings.
        features (list[str]): List of feature names.
        n_features (int): Number of features.
        encoding (str | None): Feature to use for positional encoding.
        vocabs (dict[str, Vocabulary]): Dictionary of feature vocabularies.
        embeddings (nn.ModuleDict): Module dictionary of feature embeddings.

    Methods:
        forward(**src: dict[str,torch.Tensor]) -> torch.Tensor:
            Forward pass of the module.

    """

    def __init__(
        self,
        features: list[str],
        encoding: str | None,
        vocabs: dict[str, Vocabulary],
        dim: int,
        init_std: float = 1.0,
        context_size: int = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.features = list(features)
        self.n_features = len(features)
        self.encoding = encoding
        self.vocabs = vocabs
        self.layernorm = nn.LayerNorm(dim)
        self.embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(
                    num_embeddings=vocabs[f].vocab_size,
                    embedding_dim=dim,
                    max_norm=None,
                    norm_type=2.0,
                )
                for f in features
            }
        )
        for emb in self.embeddings.values():
            emb.weight = nn.init.normal_(emb.weight, mean=0, std=init_std)

        if self.encoding == "position":
            self.embeddings["position"] = nn.Embedding(
                num_embeddings=context_size, embedding_dim=dim
            )
        elif self.encoding == "raw_time":
            raise NotImplementedError("Learned time encoding is not supported")

    def forward(self, **src: dict[str, torch.Tensor]) -> torch.Tensor:
        x = [self.embeddings[k](src[k]) for k in self.embeddings]
        x = torch.sum(torch.stack(x), dim=0)
        return self.layernorm(x)


class RotaryEmbedding(nn.Module):
    """A module that represents the rotary positional encoding for a masked language model.

    Args:
        dim (int): Dimension of the embeddings.
        freqs (list[float]): List of frequencies to use for the positional encoding.

    Attributes:
        dim (int): Dimension of the embeddings.
        freqs (torch.Tensor): Tensor of frequencies to use for the positional encoding.

    Methods:
        forward(q: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor]: Forward pass of the module.

    """

    def __init__(self, dim: int, freqs: list[float]) -> None:
        super().__init__()
        self.dim = dim
        assert len(freqs) <= dim // 2
        if len(freqs) < dim // 2:
            freqs = freqs + [0.0] * (dim // 2 - len(freqs))
        freqs = torch.tensor(freqs, dtype=torch.float64)
        self.freqs = nn.Parameter(
            torch.cat([freqs, -freqs], dim=0), requires_grad=False
        )

    @torch.no_grad()
    def _get_rotation(self, position_ids: torch.Tensor) -> tuple[torch.Tensor]:
        """Returns the rotation matrices for the given position ids."""
        assert position_ids.dtype != torch.float32
        position_ids = position_ids.to(torch.float64)
        # position_ids: [batch_size, seqlen]
        # freqs: [head_dim]
        # cos, sin: [batch_size, seq_len, head_dim]
        cos = (
            torch.cos(position_ids.unsqueeze(-1) * self.freqs)
            .requires_grad_(False)
            .to(torch.float32)
        )
        sin = (
            torch.sin(position_ids.unsqueeze(-1) * self.freqs)
            .requires_grad_(False)
            .to(torch.float32)
        )
        return cos, sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor]:
        cos, sin = self._get_rotation(p)
        # q, k: [batch_size, heads, seq_len, head_dim]
        # cos, sin: [batch_size, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class SelfAttention(nn.Module):
    """A module that represents the self-attention mechanism for a masked language model.

    Args:
        d_model (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        rotary_pos_enc (bool, optional): Whether to use rotary positional encoding. Defaults to False.
        rotary_pos_enc_freqs (list[float], optional): List of frequencies to use for the positional encoding. Defaults to None.
        biases (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Attributes:
        nhead (int): Number of attention heads.
        d_model (int): Dimension of the embeddings.
        dim_head (int): Dimension of the embeddings per head.
        scale (float): Scaling factor for the attention weights.
        rotary_pos_enc (bool): Whether to use rotary positional encoding.

    Methods:
        forward(x: torch.Tensor, p: torch.Tensor = None, return_attn_weights: bool = False) -> torch.Tensor: Forward pass of the module.
            If rotary_pos_enc is set to True, the parameter p is used for rotary positional encoding.
            If return_attn_weights is set to True, the attention weights are returned instead of the output.

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        rotary_pos_enc: bool = False,
        rotary_pos_enc_freqs: Sequence[float] | None = None,
        biases: bool = True,
    ) -> None:
        super().__init__()
        self.Wqkv = nn.Linear(d_model, d_model * 3, bias=biases)
        self.Wo = nn.Linear(d_model, d_model, bias=biases)
        self.nhead = nhead
        self.d_model = d_model
        self.dim_head = d_model // nhead
        self.scale = self.dim_head**-0.5
        self.rotary_pos_enc = rotary_pos_enc
        if rotary_pos_enc:
            self.rotary_emb = RotaryEmbedding(
                dim=self.dim_head, freqs=rotary_pos_enc_freqs
            )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, p: torch.Tensor = None, return_attn_weights: bool = False
    ) -> torch.Tensor:
        # qkv: [batch_size, seqlen, 3, nheads, headdim]
        qkv = self.Wqkv(x).view(x.size(0), x.size(1), 3, self.nhead, self.dim_head)

        # query, key, value: [batch_size, heads, seq_len, head_dim]
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        if self.rotary_pos_enc:
            query, key = self.rotary_emb(query, key, p)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scale
        attn_weights = self.softmax(attn_weights)
        if return_attn_weights:
            return attn_weights

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        out = attn_output.view(x.size(0), x.size(1), self.d_model)
        return self.Wo(out)


class EncoderMLP(nn.Module):
    """A module that represents the multi-layer perceptron for a masked language model.

    Args:
        d_model (int): Dimension of the embeddings.
        dim_feedforward (int): Dimension of the feedforward network.
        activation (Literal["relu", "gelu"], optional): The activation function to be used. Defaults to "gelu".
        gated_activation (bool, optional): Whether to use gated activation. Defaults to True.
        biases (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass of the module.

    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        activation: Literal["relu", "gelu"] = "gelu",
        gated_activation: bool = True,
        biases: bool = True,
    ) -> None:
        super().__init__()
        if gated_activation:
            self.linear1 = nn.Linear(d_model, dim_feedforward * 2, bias=biases)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=biases)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=biases)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.gated_activation = gated_activation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.gated_activation:
            x, gate = torch.chunk(x, 2, dim=-1)
            x = self.activation(x) * gate
        else:
            x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """A module that represents an encoder layer in a masked language model.

    Args:
        d_model (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        activation (Literal["relu", "gelu"], optional): The activation function to be used. Defaults to "gelu".
        gated_activation (bool, optional): Whether to use gated activation. Defaults to True.
        rotary_pos_enc (bool, optional): Whether to use rotary positional encoding. Defaults to False.
        rotary_pos_enc_freqs (list[float], optional): List of frequencies to use for the positional encoding. Defaults to None.
        biases (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Methods:
        forward(x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor: Forward pass of the module.

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: Literal["relu", "gelu"] = "gelu",
        gated_activation: bool = True,
        rotary_pos_enc: bool = False,
        rotary_pos_enc_freqs: Sequence[float] | None = None,
        biases: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = SelfAttention(
            d_model=d_model,
            nhead=nhead,
            rotary_pos_enc=rotary_pos_enc,
            rotary_pos_enc_freqs=rotary_pos_enc_freqs,
            biases=biases,
        )
        self.attn_norm = nn.LayerNorm(normalized_shape=d_model, bias=biases)
        self.mlp = EncoderMLP(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            gated_activation=gated_activation,
            biases=biases,
        )
        self.mlp_norm = nn.LayerNorm(normalized_shape=d_model, bias=biases)

    def forward(self, x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor:
        out = self.attn_norm(self.self_attn(x, p) + x)
        return self.mlp_norm(self.mlp(out) + out)


class MaskedLangModelEncoder(nn.Module):
    """A module that represents the encoder for a masked language model.

    Args:
        d_model (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        activation (Literal["relu", "gelu"], optional): The activation function to be used. Defaults to "gelu".
        gated_activation (bool, optional): Whether to use gated activation. Defaults to True.
        rotary_pos_enc (bool, optional): Whether to use rotary positional encoding. Defaults to False.
        rotary_pos_enc_freqs (list[float], optional): List of frequencies to use for the positional encoding. Defaults to None.
        biases (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Methods:
        forward(x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor: Forward pass of the module.

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        activation: Literal["relu", "gelu"] = "gelu",
        gated_activation: bool = True,
        rotary_pos_enc: bool = False,
        rotary_pos_enc_freqs: Sequence[float] | None = None,
        biases: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.modules.transformer._get_clones(
            EncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                gated_activation=gated_activation,
                rotary_pos_enc=rotary_pos_enc,
                rotary_pos_enc_freqs=rotary_pos_enc_freqs,
                biases=biases,
            ),
            num_layers,
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, p)
        return x


class MultiClassificationHead(nn.Module):
    """Module that represents the classification head for a multi-classification task.

    Args:
        d_model (int): The input dimension of the head.
        targets (list[str]): The list of target names.
        vocabs (dict[str, Vocabulary]): The dictionary of target names and their corresponding Vocabulary objects.
        bias (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Attributes:
        targets (list[str]): The list of target names.
        vocabs (dict[str, Vocabulary]): The dictionary of target names and their corresponding Vocabulary objects.
        linear (nn.ModuleList): List of linear layers for each target.

    Methods:
        forward(x: torch.Tensor) -> tuple[torch.Tensor]: Forward pass of the module.
    """

    def __init__(
        self,
        d_model: int,
        targets: list[str],
        vocabs: dict[str, Vocabulary],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.targets = targets
        self.vocabs = vocabs
        self.d_model = d_model
        self.linear = nn.ModuleDict(
            {
                t: nn.Linear(d_model, self.vocabs[t].vocab_size, bias=bias)
                for t in self.targets
            }
        )
        # set all biases to zero
        if bias:
            for linear in self.linear.values():
                nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return tuple(
            self.linear[t](x)[..., self.vocabs[t].offset :] for t in self.targets
        )


class MaskedLanguageModel(nn.ModuleDict):
    """Module that represents a masked language model.

    Args:
        params (Params): Dictionary of model parameters, for more information see the documentation in `train_mlm.py`.
        vocabs (dict[str, Vocabulary]): Dictionary of feature vocabularies.

    Methods:
        forward(**src: dict[str,torch.Tensor]) -> tuple[torch.Tensor]: Forward pass of the module.

    """

    def __init__(
        self,
        params: MaskedLangModelParams,
        vocabs: dict[str, Vocabulary],
    ) -> None:
        super().__init__(
            {
                "embedding": MaskedLangModelEmbeddingLayer(
                    features=params["features"],
                    encoding=(
                        params["encoding"]
                        if params["encoding_type"] == "learned"
                        else None
                    ),
                    vocabs=vocabs,
                    dim=params["d_model"],
                    init_std=params["emb_init_std"],
                    context_size=params["context_size"],
                ),
                "encoder": MaskedLangModelEncoder(
                    d_model=params["d_model"],
                    nhead=params["n_heads"],
                    num_layers=params["num_layers"],
                    dim_feedforward=params["dim_feedforward"],
                    activation=params["activation"],
                    gated_activation=params["gated_activation"],
                    rotary_pos_enc=params["encoding_type"] == "rotary",
                    rotary_pos_enc_freqs=(
                        params["encoding_freqs"]
                        if params["encoding_type"] == "rotary"
                        else None
                    ),
                    biases=params["biases"],
                ),
                "head": MultiClassificationHead(
                    d_model=params["d_model"],
                    targets=params["targets"],
                    vocabs=vocabs,
                    bias=params["head_bias"],
                ),
            }
        )

        self.rotary_pos_enc = params["encoding_type"] == "rotary"
        self.encoding = params["encoding"]

        # tie weigths of the embedding layer and the output layer
        if params["tie_weights"]:
            for t in params["targets"]:
                self["head"].linear[t].weight = self["embedding"].embeddings[t].weight

    def forward(self, **src: dict[str, torch.Tensor]) -> tuple[torch.Tensor]:
        x = self.embedding(**src)
        x = self.encoder(x, p=src[self.encoding] if self.rotary_pos_enc else None)
        return self.head(x)


class MultiTargetLoss(nn.Module):
    """A module that combines multiple loss functions for multi-target learning.

    Args:
        loss_fns (list[nn.Module]): A list of loss functions to be combined.

    Methods:
        forward(output: tuple[torch.tensor], target: tuple[torch.tensor]) -> torch.tensor: Forward pass of the module.

    """

    def __init__(self, loss_fns: list[nn.Module]) -> None:
        super().__init__()
        self.loss_fns = nn.ModuleList(loss_fns)

    def forward(
        self, output: tuple[torch.tensor], target: tuple[torch.tensor]
    ) -> torch.tensor:
        return torch.mean(
            torch.stack(
                tuple(
                    loss_fn(out, tar)
                    for loss_fn, out, tar in zip(self.loss_fns, output, target)
                )
            )
        )


# wrappers to make the models work with TensorDicts


class MaskedLangModelInferenceWrapper(TensorDictModule):
    """A wrapper class for performing inference using a MaskedLanguageModel and TensorDicts.

    Args:
        model (MaskedLanguageModel): The masked language model to be used for inference.
        layers (list[str], optional): A list of layer names to be used from the model.
                                      Defaults to ["embedding", "encoder"].

    """

    def __init__(
        self,
        model: MaskedLanguageModel,
        layers: Sequence[str] = ("embedding", "encoder"),
    ) -> None:
        super().__init__(
            nn.Sequential(OrderedDict({layer: model[layer] for layer in layers})),
            in_keys={f: f for f in model["embedding"].features + [model.encoding]},
            out_keys=["output"],
        )
        self.encoding = model.encoding
        self.module.forward = self._star_forward

    @torch.no_grad()
    def _star_forward(self, **x: dict[str, torch.Tensor]) -> torch.Tensor:
        """This function is used to override the forward method of the nn.Sequential module to make it compatible with varying numbers of input arguments."""
        p = x.get(self.encoding, None)
        x = self.module[0](**x)
        for module in self.module[1:]:
            x = module(x, p)
        return x


class MaskedLangModelAttentionWrapper(MaskedLangModelInferenceWrapper):
    """Wrapper class for extracting attention weights from a MaskedLanguageModel.

    Args:
        model (MaskedLanguageModel): The masked language model to be used for extracting attention weights.

    Methods:
        forward(batch: TensorDict[str, torch.Tensor]) -> TensorDict[str, torch.Tensor]:
            Performs forward pass of the model on the given batch and extracts attention weights from the encoder layers.

    """

    def __init__(self, model: MaskedLanguageModel) -> None:
        super().__init__(model, layers=["embedding"])
        self.encoder = model["encoder"].layers

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        batch = super().forward(batch)
        attention = []
        for module in self.encoder:
            attention.append(
                module.self_attn(
                    x=batch["output"],
                    p=batch[self.encoding],
                    return_attn_weights=True,
                )
            )
            batch["output"] = module(batch["output"], batch[self.encoding])
        batch["attention"] = torch.stack(attention)
        return batch


class MaskedLangModelEvalWrapper(TensorDictModule):
    """Wrapper class for evaluating MaskedLanguageModel with TensorDicts.

    Args:
        model (MaskedLanguageModel): The masked language model to be evaluated.

    Attributes:
        model (MaskedLanguageModel): The masked language model being evaluated.
        true_target_keys (List[str]): The keys for the true target values.
        masked_out_keys (List[str]): The keys for the masked output values.

    Methods:
        forward(batch: TensorDict[str, torch.Tensor]) -> TensorDict[str, torch.Tensor]:
            Performs forward pass of the model on the given batch.

    """

    def __init__(self, model: MaskedLanguageModel) -> None:
        super().__init__(
            model,
            in_keys={f"{f}_mask": f for f in model["embedding"].features}
            | {model.encoding: model.encoding},
            out_keys=[f"{t}_pred" for t in model["head"].targets],
        )
        self.true_target_keys = [f"{t}_true" for t in model["head"].targets]
        self.masked_out_keys = [f"{k}_mask" for k in self.out_keys]

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        mask_idx = torch.unbind(batch["mask_index"])
        batch = super().forward(batch)
        for t, k in zip(self.module["head"].targets, self.true_target_keys):
            batch[k] = self.module["head"].vocabs[t].compute_targets(batch[t][mask_idx])
        for k, m in zip(self.out_keys, self.masked_out_keys):
            batch[m] = batch[k][mask_idx]
        return batch


class MaskedLangModelTrainWrapper(MaskedLangModelEvalWrapper):
    """Wrapper class for training MaskedLanguageModel with TensorDicts.

    Args:
        model (MaskedLanguageModel): The masked language model.
        loss_fn (MultiTargetLoss): The loss function.

    """

    def __init__(self, model: MaskedLanguageModel, loss_fn: MultiTargetLoss) -> None:
        super().__init__(model)
        self.loss_fn = loss_fn

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        batch = super().forward(batch)
        batch["loss"] = self.loss_fn(
            tuple(batch[k] for k in self.masked_out_keys),
            tuple(batch[t] for t in self.true_target_keys),
        )
        return batch


# token clustering models
