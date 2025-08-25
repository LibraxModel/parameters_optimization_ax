from gpytorch.kernels import MaternKernel
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from ax.utils.stats.model_fit_stats import MSE
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec, ModelConfig

surrogate_spec = SurrogateSpec(
    model_configs=[
        # Select between two models:
        # An additive mixture of relatively strong SAAS priors with input Warping.
        # A relatively vanilla GP with a Matern kernel.
        ModelConfig(
            botorch_model_class=AdditiveMapSaasSingleTaskGP,
            input_transform_classes=[Warp],
            # Additional options for the model constructor. These need to be supported
            # by the input constructor. We will see that below.
            model_options={},
        ),
        ModelConfig(
            botorch_model_class=SingleTaskGP,
            covar_module_class=MaternKernel,
            covar_module_options={"nu": 2.5},
        ),
    ],
    eval_criterion=MSE,  # Select the model to use as the one that minimizes mean squared error.
    allow_batched_models=False,  # Forces each metric to be modeled with an independent BoTorch model.
    # If we wanted to specify different options for different metrics.
    # metric_to_model_configs: dict[str, list[ModelConfig]]
)


from typing import Optional

from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.datasets import SupervisedDataset
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from torch import Tensor


class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, train_Yvar: Optional[Tensor] = None):
        # NOTE: This ignores train_Yvar and uses inferred noise instead.
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        # Depending on the experiment setup, additional arguments may be passed in here.
    ) -> dict[str, Tensor]:
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "train_Yvar": training_data.Yvar,
        }

from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.modelbridge.registry import Generators


def construct_generation_strategy(
    generator_spec: GeneratorSpec, node_name: str,
) -> GenerationStrategy:
    """Constructs a Center + Sobol + Modular BoTorch `GenerationStrategy`
    using the provided `generator_spec` for the Modular BoTorch node.
    """
    botorch_node = GenerationNode(
        node_name=node_name,
        generator_specs=[generator_spec],
    )
    sobol_node = GenerationNode(
        node_name="Sobol",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.SOBOL,
                # Let's use model_kwargs to set the random seed.
                model_kwargs={"seed": 0},
            ),
        ],
        transition_criteria=[
            # Transition to BoTorch node once there are 5 trials on the experiment.
            MinTrials(
                threshold=5,
                transition_to=botorch_node.node_name,
                use_all_trials_in_exp=True,
            )
        ]
    )
    # Center node is a customized node that uses a simplified logic and has a
    # built-in transition criteria that transitions after generating once.
    center_node = CenterGenerationNode(next_node_name=sobol_node.node_name)
    return GenerationStrategy(
        name=f"Center+Sobol+{node_name}",
        nodes=[center_node, sobol_node, botorch_node]
    )

# Let's construct the simplest version with all defaults.
construct_generation_strategy(
    generator_spec=GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR),
    node_name="Modular BoTorch",
)