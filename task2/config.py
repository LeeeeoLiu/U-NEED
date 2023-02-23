import os
import argparse

from utils.io_utils import load_json, save_json, get_or_create_logger


CONFIGURATION_FILE_NAME = "run_config.json"

logger = get_or_create_logger(__name__)


def add_config(parser):
    """ define arguments """
    group = parser.add_argument_group("Construction")
    group.add_argument("-backbone", type=str, default="t5-small")
    group.add_argument("-task", type=str, default="e2e",
                       choices=["dst", "e2e"])
    group.add_argument("-add_auxiliary_task", action="store_true")
    group.add_argument("-context_size", type=int, default=-1)
    group.add_argument("-ururu", action="store_true")

    group = parser.add_argument_group("Training")
    group.add_argument("-batch_size", type=int, default=8)
    group.add_argument("-epochs", type=int, default=10)
    group.add_argument("-pre_epoch", type=int, default=10)
    group.add_argument("-warmup_steps", type=int, default=-1)
    group.add_argument("-warmup_ratio", type=float, default=0.2)
    group.add_argument("-learning_rate", type=float, default=5e-4)
    group.add_argument("-weight_decay", type=float, default=0.0)
    group.add_argument("-grad_accum_steps", type=int, default=1)
    group.add_argument("-max_grad_norm", type=float, default=1.0)
    group.add_argument("-aux_loss_coeff", type=float, default=0.5)
    group.add_argument("-resp_loss_coeff", type=float, default=1.0)
    group.add_argument("-dscl_loss_coeff", type=float, default=1.0)
    group.add_argument("-ascl_loss_coeff", type=float, default=0.1)
    group.add_argument("-num_train_dialogs", type=int, default=-1)
    group.add_argument("-train_from", type=str, default=None)
    group.add_argument("-no_validation", action="store_true")
    group.add_argument("-no_learning_rate_decay", action="store_true")
    
    group = parser.add_argument_group("Prediction")
    group.add_argument("-pred_data_type", type=str, default="test",
                       choices=["dev", "test"])
    group.add_argument("-overwrite_with_span", action="store_true")
    group.add_argument("-beam_size", type=int, default=1)
    group.add_argument("-do_sample", action="store_true")
    group.add_argument("-top_k", type=int, default=0)
    group.add_argument("-top_p", type=float, default=0.7)
    group.add_argument("-temperature", type=float, default=1.0)
    group.add_argument("-use_true_dbpn", action="store_true")
    group.add_argument("-use_true_curr_aspn", action="store_true")
    group.add_argument("-use_true_prev_bspn", action="store_true")
    group.add_argument("-use_true_prev_aspn", action="store_true")
    group.add_argument("-use_true_prev_resp", action="store_true")
    group.add_argument("-top_n", type=int, default=5)
    group.add_argument("-output", type=str, default="output")

    group = parser.add_argument_group("Misc")
    group.add_argument("-run_type", type=str, required=True,
                       choices=["train", "predict"])
    group.add_argument("-excluded_domains", type=str, nargs="+")
    group.add_argument("-model_dir", type=str, default="checkpoints")
    group.add_argument("-seed", type=int, default=42)
    group.add_argument("-ckpt", type=str, default=None)
    group.add_argument("-log_frequency", type=int, default=100)
    group.add_argument("-max_to_keep_ckpt", type=int, default=10)
    group.add_argument("-num_gpus", type=int, default=1)
    group.add_argument("-add_ds_contrastive_task", action="store_true")
    group.add_argument("-add_as_contrastive_task", action="store_true")
    group.add_argument("-group_wise", action="store_true")
    group.add_argument("-point_wise", action="store_true")    
    group.add_argument("-dsas_mlp_shared", action="store_true")
    group.add_argument("-ds_mlp_shared", action="store_true")
    group.add_argument("-as_mlp_shared", action="store_true")
    group.add_argument("-ds_mlp", action="store_true")
    group.add_argument("-as_mlp", action="store_true")
    group.add_argument("-T", type=float, default=0.5)
    
def get_config():
    """ return ArgumentParser Instance """
    parser = argparse.ArgumentParser(
        description="Configuration of task-oriented dialogue model with multi-task learning.")

    add_config(parser)

    return parser


if __name__ == "__main__":
    configs = get_config()
