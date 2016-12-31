from .nbutils import strip_consts, show_graph
from .nputils import log_sum_exp as np_log_sum_exp, kl_normal
from .utils import progbar
from .tbutils import (
    log_sum_exp as tb_log_sum_exp, cross_entropy_with_logits, ones_initializer,
    zeros_initializer, assign_moving_average
)
