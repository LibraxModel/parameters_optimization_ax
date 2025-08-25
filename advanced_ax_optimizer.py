# advanced_ax_optimizer.py  ——  for ax-platform 1.0.0
from __future__ import annotations
from typing import Any, Dict, List, Optional
import importlib, random

import numpy as np
import torch

# Ax 1.0.0 public imports
from ax.generation_strategy.generation_strategy import GenerationStrategy, GenerationStep     #  [oai_citation:2‡GitHub](https://github.com/facebook/Ax/blob/master/ax/modelbridge/registry.py?utm_source=chatgpt.com)
from ax.modelbridge.registry import Generators                                              # ← 新注册表  [oai_citation:3‡ax.dev](https://ax.dev/docs/0.5.0/tutorials/generation_strategy/?utm_source=chatgpt.com)
from ax.models.torch.botorch_modular.model import BoTorchGenerator
from ax.models.torch.botorch_modular.utils import ModelConfig
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.service.ax_client import AxClient, ObjectiveProperties
from botorch.acquisition.acquisition import AcquisitionFunction

from ax_optimizer import BayesianOptimizer       # 你先前的默认优化器

# ---------- 辅助函数 ----------
def _str_to_cls(name: str, paths: List[str]):
    """把字符串转成类对象，搜索多个模块路径。"""
    for p in paths:
        try:
            m = importlib.import_module(p)
            if hasattr(m, name):
                return getattr(m, name)
        except ModuleNotFoundError:
            continue
    raise ValueError(f"{name} not found in {paths}")

def _resolve_acqf(name: str) -> type[AcquisitionFunction]:
    return _str_to_cls(name, [
        "botorch.acquisition.monte_carlo",
        "botorch.acquisition.analytic",
        "botorch.acquisition.logei",
        "botorch.acquisition.multi_objective.monte_carlo",
    ])                                                                    #  [oai_citation:4‡botorch.readthedocs.io](https://botorch.readthedocs.io/en/stable/acquisition.html?utm_source=chatgpt.com)

# ---------- 主类 ----------
class AdvancedBayesianOptimizer(BayesianOptimizer):
    def __init__(
        self,
        search_space: List[Dict[str, Any]],
        optimization_config: Dict[str, Any],
        *,
        surrogate_model: str = "SingleTaskGP",
        kernel: str = "RBFKernel",
        likelihood: str = "GaussianLikelihood",
        acquisition: str = "qExpectedImprovement",
        acquisition_options: Optional[Dict[str, Any]] = None,
        init_strategy: str = "Sobol",  # Sobol | Thompson | Uniform
        n_init: int = 0,
        q_batch: int = 1,
        max_parallelism: Optional[int] = None,
        num_folds: int = 5,
        use_posterior_predictive: bool = False,
        eval_metric: str = "RANK_CORRELATION",
        seed: int = 0,
        experiment_name: str = "advanced_bo",
    ):
        # ---- 随机种子 ----
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                        
        # ---- 解析类对象 ----
        botorch_model_cls = _str_to_cls(
            surrogate_model,
            ["botorch.models", "botorch.models.gp_regression",
             "botorch.models.gp_regression_fidelity",
             "botorch.models.fully_bayesian", "botorch.models.contextual"]
        )
        kernel_cls = _str_to_cls(
            kernel,
            ["gpytorch.kernels", "ax.models.torch.botorch_modular.kernels"] # ScaleMaternKernel 所在模块  [oai_citation:6‡qmcpy.readthedocs.io](https://qmcpy.readthedocs.io/en/sphinx/demo_rst/digital_net_b2.html?utm_source=chatgpt.com)
        )
        likelihood_cls = _str_to_cls(likelihood, ["gpytorch.likelihoods"])

        # ---- SurrogateSpec ----
        model_cfg = ModelConfig(
            name="user_surrogate",
            botorch_model_class=botorch_model_cls,
            covar_module_class=kernel_cls,
            likelihood_class=likelihood_cls,
        )
        surrogate_spec = SurrogateSpec(model_configs=[model_cfg])

        # ---- Acquisition ----
        acqf_class = _resolve_acqf(acquisition)

        botorch_gen = BoTorchGenerator(
            surrogate_spec=surrogate_spec,
            botorch_acqf_class=acqf_class,
            acquisition_options=acquisition_options or {},
        )

        # ---- GenerationStrategy ----
        init_map = {
            "Sobol":        Generators.SOBOL,            # 新注册表名   [oai_citation:7‡ax.dev](https://ax.dev/docs/tutorials/modular_botorch/?utm_source=chatgpt.com)
            "Thompson":     Generators.THOMPSON,   # 同理
            "Uniform":      Generators.UNIFORM,
        }
        gs = GenerationStrategy(
            name="ADV_STRATEGY",
            steps=[
                GenerationStep(
                    model=init_map[init_strategy],
                    num_trials=n_init,
                    max_parallelism=n_init,
                ),
                GenerationStep(
                    model=Generators.BOTORCH_MODULAR,
                    num_trials=-1,
                    max_parallelism=max_parallelism or q_batch,
                    model_gen_kwargs={"n": q_batch},
                ),
            ],
        )

        # ---- AxClient ----
        objective = {
            optimization_config["objective_name"]: ObjectiveProperties(
                minimize=optimization_config.get("minimize", True)
            )
        }
        self.ax_client = AxClient(random_seed=seed, generation_strategy=gs)
        self.ax_client.create_experiment(
            name=experiment_name,
            parameters=search_space,
            objectives=objective,
            tracking_metric_names=optimization_config.get("additional_metrics", []),
        )

        # —— 记录 ——  
        self.experiment_name = experiment_name
        self.minimize = optimization_config.get("minimize", True)
        self.random_seed = seed
        self.trial_count = 0
        self._cv_num_folds = num_folds
        self._cv_use_pp = use_posterior_predictive
        self._eval_metric = eval_metric

    # ---- 交叉验证 + 指标 ----
    def cross_validate(self) -> float:
        means, _ = self.ax_client.cross_validate(
            num_folds=self._cv_num_folds,
            use_posterior_predictive=self._cv_use_pp,
        )
        y_true, y_pred = means["observed"].values, means["predicted"].values
        if self._eval_metric == "RANK_CORRELATION":
            from scipy.stats import kendalltau
            return float(kendalltau(y_true, y_pred)[0])
        if self._eval_metric == "MSE":
            return float(np.mean((y_true - y_pred) ** 2))
        if self._eval_metric == "MAPE":
            from sklearn.metrics import mean_absolute_percentage_error
            return float(mean_absolute_percentage_error(y_true, y_pred))
        raise ValueError(f"Unknown eval metric {self._eval_metric}")

# ───────── quick smoke-test ─────────
if __name__ == "__main__":
    def simulated_experiment(x: float) -> float:
        """示例目标函数：在 x≈0.3 处取得最小值，带 0.01 方差的噪声。"""
        noise = random.gauss(0, 0.01)
        return (x - 0.3) ** 2 + noise

    # ---------- 2. 构建优化器 ----------
    search_space = [
        {"name": "x", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"},
    ]
    conf = {"objective_name": "y", "minimize": True}

    bo = AdvancedBayesianOptimizer(
        search_space=search_space,
        optimization_config=conf,
        surrogate_model="SingleTaskGP",
        kernel="ScaleMaternKernel",
        acquisition="qUpperConfidenceBound",
        acquisition_options={"beta": 1.5},
        n_init=4,          
        q_batch=4,        
        seed=2025,
    )

    # ---------- 第 1 步：一次性生成并完成 4 个 Sobol ----------
    init_trials = bo.get_next_parameters(n=4)        # 生成 4 组
    for p, tid in init_trials:
        bo.update_experiment(tid, {"y": simulated_experiment(**p)})

    # ---------- 第 2 步：再请求 1 批 → 此时已进入 q-UCB 阶段 ----------
    bayes_trials = bo.get_next_parameters(n=4)       # 触发 BoTorch
    for p, tid in bayes_trials:
        bo.update_experiment(tid, {"y": simulated_experiment(**p)})

    print("当前最佳参数:", bo.get_best_parameters()[0])