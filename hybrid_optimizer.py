"""
融合优化策略：结合 GP 贝叶斯优化和 LLM 推荐
使用 GP 的采集函数给 LLM 推荐的点打分，只接受"不过分差"的点
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ax_optimizer import BayesianOptimizer, ExperimentResult
from LLINBO_agent import (
    LLINBOAgent, ProblemContext, Parameter, PriorExperiment, LLMConfig
)


class HybridOptimizer:
    """
    融合优化器：结合贝叶斯优化（GP）和大模型（LLM）的推荐策略
    
    策略：
    1. 使用 LLM 生成推荐点
    2. 使用 GP 的采集函数评估这些点
    3. 获取 GP 推荐的最佳点及其采集函数值
    4. 对于每个 LLM 点，如果 (最佳点的采集函数值 - LLM点的采集函数值) < 阈值，则接受
    """
    
    def __init__(
        self,
        # 参数空间定义（用于 LLM）
        llm_parameters: List[Parameter],
        # 参数空间定义（用于 GP，Ax 格式）
        gp_search_space: List[Dict[str, Any]],
        # 优化配置
        optimization_config: Dict[str, Any],
        # 问题背景（用于 LLM）
        problem_context: ProblemContext,
        # LLM 配置
        llm_config: Optional[LLMConfig] = None,
        # GP 优化器配置
        gp_experiment_name: str = "hybrid_optimization",
        gp_random_seed: Optional[int] = None,
        gp_surrogate_model_class: Optional[Any] = None,
        gp_kernel_class: Optional[Any] = None,
        gp_kernel_options: Optional[Dict[str, Any]] = None,
        gp_acquisition_function_class: Optional[Any] = None,
        gp_acquisition_function_options: Optional[Dict[str, Any]] = None,
        # 融合策略参数
        acquisition_threshold: float = 0.1,  # 采集函数差值阈值（固定阈值，当 use_dynamic_threshold=False 时使用）
        use_dynamic_threshold: bool = True,  # 是否使用基于方差的动态阈值
        threshold_multiplier: float = 1.0,  # 动态阈值倍数（阈值 = threshold_multiplier * 预测标准差）
        random_seed: Optional[int] = None
    ):
        """
        初始化融合优化器
        
        Args:
            llm_parameters: LLM 参数空间定义（Parameter 列表）
            gp_search_space: GP 参数空间定义（Ax 格式）
            optimization_config: 优化配置
            problem_context: 问题背景（用于 LLM）
            llm_config: LLM 配置
            gp_experiment_name: GP 实验名称
            gp_random_seed: GP 随机种子
            gp_surrogate_model_class: GP 代理模型类
            gp_kernel_class: GP 核函数类
            gp_kernel_options: GP 核函数选项
            gp_acquisition_function_class: GP 采集函数类
            gp_acquisition_function_options: GP 采集函数选项
            acquisition_threshold: 采集函数差值阈值（默认 0.1，当 use_dynamic_threshold=False 时使用）
            use_dynamic_threshold: 是否使用基于方差的动态阈值（默认 True）
            threshold_multiplier: 动态阈值倍数（默认 1.0，阈值 = threshold_multiplier * 预测标准差）
            random_seed: 随机种子
        """
        self.acquisition_threshold = acquisition_threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        self.threshold_multiplier = threshold_multiplier
        self.random_seed = random_seed
        
        # 初始化 LLM Agent
        self.llm_agent = LLINBOAgent(
            problem_context=problem_context,
            parameters=llm_parameters,
            objectives=optimization_config.get("objectives", {}),
            llm_config=llm_config,
            prior_experiments=None,  # 先验数据将通过 add_prior_experiments 添加
            random_seed=random_seed
        )
        
        # 初始化 GP 优化器
        self.gp_optimizer = BayesianOptimizer(
            search_space=gp_search_space,
            optimization_config=optimization_config,
            experiment_name=gp_experiment_name,
            random_seed=gp_random_seed,
            surrogate_model_class=gp_surrogate_model_class,
            kernel_class=gp_kernel_class,
            kernel_options=gp_kernel_options,
            acquisition_function_class=gp_acquisition_function_class,
            acquisition_function_options=gp_acquisition_function_options
        )
        
        # 记录融合历史
        self.hybrid_history: List[Dict[str, Any]] = []
    
    def add_prior_experiments(self, experiments: List[ExperimentResult]) -> None:
        """
        添加先验实验数据到 LLM 和 GP
        
        Args:
            experiments: 先验实验结果列表
        """
        # 添加到 GP
        self.gp_optimizer.add_prior_experiments(experiments)
        
        # 添加到LLM（需要检查重复，避免重复添加）
        # 获取现有参数的唯一标识
        existing_params = set()
        if self.llm_agent.prior_experiments:
            for exp in self.llm_agent.prior_experiments:
                param_key = tuple(sorted(exp.parameters.items()))
                existing_params.add(param_key)
        
        # 只添加不重复的新数据
        for experiment in experiments:
            param_key = tuple(sorted(experiment.parameters.items()))
            if param_key not in existing_params:
                self.llm_agent.add_experiment_result(
                    parameters=experiment.parameters,
                    metrics=experiment.metrics,
                    metadata=experiment.metadata
                )
                existing_params.add(param_key)
    
    def _evaluate_acquisition_value(self, parameters: Dict[str, Any]) -> float:
        """
        评估给定参数点的采集函数值
        
        使用 Ax 的 TorchAdapter.evaluate_acquisition_function 方法精确评估采集函数值
        
        Args:
            parameters: 参数配置字典
            
        Returns:
            acquisition_value: 采集函数值
        """
        try:
            from ax.core.observation import ObservationFeatures
            
            # 创建观察特征
            obsf = ObservationFeatures(parameters=parameters)
            
            # 获取 GenerationStrategy 的 adapter
            gs = self.gp_optimizer.ax_client.generation_strategy
            adapter = gs.adapter
            
            if adapter is None:
                # 如果还没有 adapter，返回一个默认值
                print("⚠️ 警告: GenerationStrategy 还没有 adapter，无法评估采集函数值")
                return 0.0
            
            # 使用 TorchAdapter 的 evaluate_acquisition_function 方法
            # 这是 Ax 提供的精确评估采集函数值的 API
            try:
                acqf_values = adapter.evaluate_acquisition_function(
                    observation_features=[obsf],
                    search_space=self.gp_optimizer.ax_client.experiment.search_space,
                    optimization_config=self.gp_optimizer.ax_client.experiment.optimization_config,
                    pending_observations=None,
                    fixed_features=None,
                    acq_options=None
                )
                
                if acqf_values and len(acqf_values) > 0:
                    return float(acqf_values[0])
                else:
                    return 0.0
                    
            except Exception as e:
                print(f"⚠️ 使用 TorchAdapter.evaluate_acquisition_function 失败: {e}")
                # 如果失败，返回默认值
                return 0.0
                
        except Exception as e:
            print(f"⚠️ 评估采集函数值时发生错误: {e}")
            return 0.0
    
    def _get_prediction_std(self, parameters: Dict[str, Any]) -> float:
        """
        获取给定参数点的预测标准差（归一化后的平均变异系数）
        
        使用 Ax 的 get_model_predictions 方法获取高斯过程的预测标准差
        对于多目标优化，使用变异系数（CV = std/mean）来消除量纲影响
        
        Args:
            parameters: 参数配置字典
            
        Returns:
            prediction_std: 预测标准差（如果多目标，返回平均变异系数，无量纲）
        """
        try:
            # 使用 Ax 的 get_model_predictions 方法获取预测
            predictions_dict = self.gp_optimizer.ax_client.get_model_predictions(
                metric_names=None,  # 获取所有指标
                parameterizations={0: parameters}
            )
            
            if not predictions_dict or 0 not in predictions_dict:
                return 0.0
            
            # 获取所有指标的预测
            metric_predictions = predictions_dict[0]
            
            # 计算变异系数（CV = std/mean），然后平均
            # 变异系数是无量纲的，可以用于不同量纲目标的比较
            cv_values = []
            for metric_name, (mean, sem) in metric_predictions.items():
                if abs(mean) > 1e-10:  # 避免除零
                    cv = abs(sem / mean)  # 变异系数（无量纲）
                    cv_values.append(cv)
                else:
                    # 如果均值接近0，使用绝对标准差
                    cv_values.append(abs(sem))
            
            if cv_values:
                # 返回平均变异系数
                avg_cv = sum(cv_values) / len(cv_values)
                return avg_cv
            else:
                return 0.0
                
        except Exception as e:
            print(f"⚠️ 获取预测标准差失败: {e}")
            return 0.0
    
    def _get_gp_best_acquisition_value(self) -> Tuple[Dict[str, Any], float, float]:
        """
        获取 GP 推荐的最佳点及其采集函数值和预测标准差
        
        Returns:
            best_parameters: 最佳参数配置
            best_acquisition_value: 最佳采集函数值
            best_prediction_std: 最佳点的预测标准差
        """
        try:
            # 获取 GP 推荐的最佳点
            gp_trials, _ = self.gp_optimizer.ax_client.get_next_trials(max_trials=1)
            
            if not gp_trials:
                return {}, 0.0, 0.0
            
            # 获取第一个（最佳）推荐点
            best_trial_index = list(gp_trials.keys())[0]
            best_parameters = gp_trials[best_trial_index]
            
            # 评估该点的采集函数值
            best_acquisition_value = self._evaluate_acquisition_value(best_parameters)
            
            # 获取该点的预测标准差
            best_prediction_std = self._get_prediction_std(best_parameters)
            
            return best_parameters, best_acquisition_value, best_prediction_std
            
        except Exception as e:
            print(f"⚠️ 获取 GP 最佳点失败: {e}")
            return {}, 0.0, 0.0
    
    def suggest_parameters(
        self,
        num_suggestions: int = 1,
        use_llm: bool = True,
        use_gp: bool = True,
        print_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        生成融合推荐参数
        
        Args:
            num_suggestions: 需要生成的建议数量
            use_llm: 是否使用 LLM 推荐
            use_gp: 是否使用 GP 推荐
            print_details: 是否打印详细信息
            
        Returns:
            accepted_suggestions: 被接受的推荐参数列表
        """
        accepted_suggestions = []
        
        # 1. 获取 GP 推荐的最佳点及其采集函数值和预测标准差
        gp_best_params, gp_best_acq_value, gp_best_std = self._get_gp_best_acquisition_value()
        
        if print_details:
            print(f"\n📊 GP 推荐最佳点:")
            print(f"   参数: {gp_best_params}")
            print(f"   采集函数值: {gp_best_acq_value:.6f}")
            print(f"   预测标准差: {gp_best_std:.6f}")
        
        # 2. 使用 LLM 生成推荐点
        llm_suggestions = []
        if use_llm:
            try:
                llm_suggestions = self.llm_agent.suggest_parameters(
                    num_suggestions=num_suggestions * 2,  # 生成更多候选点
                    print_prompt=True,
                    print_response=True
                )
                if print_details:
                    print(f"\n🤖 LLM 生成了 {len(llm_suggestions)} 个推荐点")
            except Exception as e:
                print(f"⚠️ LLM 推荐失败: {e}")
        
        # 3. 使用 GP 采集函数评估 LLM 推荐的点
        evaluated_llm_suggestions = []
        for llm_params in llm_suggestions:
            try:
                acq_value = self._evaluate_acquisition_value(llm_params)
                
                # 获取该点的预测标准差
                prediction_std = self._get_prediction_std(llm_params)
                
                # 计算与最佳点的差值
                acq_diff = gp_best_acq_value - acq_value
                
                # 计算动态阈值（基于该点的预测标准差）
                if self.use_dynamic_threshold:
                    # 使用该点的预测标准差作为阈值
                    dynamic_threshold = self.threshold_multiplier * prediction_std
                    # 如果标准差为0或太小，使用固定阈值作为下限
                    if dynamic_threshold < self.acquisition_threshold:
                        dynamic_threshold = self.acquisition_threshold
                else:
                    # 使用固定阈值
                    dynamic_threshold = self.acquisition_threshold
                
                evaluated_llm_suggestions.append({
                    "parameters": llm_params,
                    "acquisition_value": acq_value,
                    "acquisition_diff": acq_diff,
                    "prediction_std": prediction_std,
                    "threshold": dynamic_threshold,
                    "source": "LLM"
                })
                
                if print_details:
                    print(f"\n   LLM 推荐点: {llm_params}")
                    print(f"   采集函数值: {acq_value:.6f}")
                    print(f"   预测标准差: {prediction_std:.6f}")
                    print(f"   与最佳点差值: {acq_diff:.6f}")
                    print(f"   动态阈值: {dynamic_threshold:.6f} ({'基于方差' if self.use_dynamic_threshold else '固定'})")
                    print(f"   是否接受: {'✅' if acq_diff < dynamic_threshold else '❌'}")
                    
            except Exception as e:
                print(f"⚠️ 评估 LLM 推荐点失败: {e}")
        
        # 4. 筛选"不过分差"的 LLM 推荐点（使用动态阈值）
        # 特殊情况：如果 GP 最佳点的采集函数值为负，说明 GP 对当前搜索空间信心不足
        # 此时应该更信任 LLM 的推荐，直接采纳前 n 个 LLM 推荐的点
        if gp_best_acq_value < -0.5:
            if print_details:
                print(f"\n⚠️ GP 最佳点的采集函数值为负 ({gp_best_acq_value:.6f})，说明 GP 对当前搜索空间信心不足")
                print(f"   采用特殊策略：直接采纳前 {num_suggestions} 个 LLM 推荐点（按大模型推荐信心从高到低排序）")
            
            # 直接推荐前num_suggestions个LLM推荐的点
            valid_suggestions = evaluated_llm_suggestions[:num_suggestions]
            
        else:
            # 正常情况：使用阈值筛选
            # 先筛选出所有满足阈值的点
            valid_suggestions = []
            for llm_suggestion in evaluated_llm_suggestions:
                if llm_suggestion["acquisition_diff"] < llm_suggestion["threshold"]:
                    valid_suggestions.append(llm_suggestion)
            
            # 如果满足阈值的点超过 num_suggestions 个，按采集函数差值排序，优先选择差值更小的点
            if len(valid_suggestions) > num_suggestions:
                # 按采集函数差值从小到大排序（差值越小，说明越接近最佳点）
                valid_suggestions.sort(key=lambda x: x["acquisition_diff"])
                if print_details:
                    print(f"\n📊 满足阈值的点有 {len(valid_suggestions)} 个，按采集函数差值排序后选择前 {num_suggestions} 个")
                    print(f"   排序后的前 {num_suggestions} 个点的采集函数差值:")
                    for i, suggestion in enumerate(valid_suggestions[:num_suggestions], 1):
                        print(f"   {i}. 差值: {suggestion['acquisition_diff']:.6f}, 阈值: {suggestion['threshold']:.6f}")
                # 只保留前 num_suggestions 个
                valid_suggestions = valid_suggestions[:num_suggestions]
        
        # 将筛选后的点添加到接受列表
        for suggestion in valid_suggestions:
            accepted_suggestions.append(suggestion["parameters"])
        
        # 5. 如果还需要更多推荐，使用 GP 推荐补充
        if use_gp and len(accepted_suggestions) < num_suggestions:
            try:
                gp_trials = self.gp_optimizer.get_next_parameters(
                    n=num_suggestions - len(accepted_suggestions)
                )
                for trial_index, gp_params in gp_trials:
                    # 检查是否与已接受的 LLM 推荐重复
                    is_duplicate = False
                    for accepted in accepted_suggestions:
                        if self._parameters_equal(accepted, gp_params):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        accepted_suggestions.append(gp_params)
                        if print_details:
                            print(f"\n   GP 补充推荐点: {gp_params}")
                            
            except Exception as e:
                print(f"⚠️ GP 推荐失败: {e}")
        
        # 6. 记录融合历史
        self.hybrid_history.append({
            "gp_best_params": gp_best_params,
            "gp_best_acq_value": gp_best_acq_value,
            "gp_best_std": gp_best_std,
            "llm_suggestions_count": len(llm_suggestions),
            "accepted_count": len(accepted_suggestions),
            "evaluated_llm_suggestions": evaluated_llm_suggestions,
            "use_dynamic_threshold": self.use_dynamic_threshold
        })
        
        if print_details:
            print(f"\n✅ 融合推荐完成:")
            print(f"   LLM 推荐数: {len(llm_suggestions)}")
            print(f"   接受数: {len(accepted_suggestions)}")
            print(f"   最终推荐数: {len(accepted_suggestions)}")
        
        return accepted_suggestions[:num_suggestions]
    
    def _parameters_equal(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        """检查两个参数配置是否相等"""
        if set(params1.keys()) != set(params2.keys()):
            return False
        
        for key in params1.keys():
            val1 = params1[key]
            val2 = params2[key]
            
            # 对于浮点数，使用容差比较
            if isinstance(val1, float) or isinstance(val2, float):
                if abs(float(val1) - float(val2)) > 1e-6:
                    return False
            else:
                if val1 != val2:
                    return False
        
        return True
    
    def update_experiment(
        self,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        更新实验结果
        
        Args:
            parameters: 参数配置
            metrics: 指标结果
            metadata: 额外元数据
        """
        # 更新到 GP
        # 需要先找到对应的 trial_index
        # 这里简化处理：创建一个新的 trial
        trial_index, _ = self.gp_optimizer.ax_client.attach_trial(parameters)
        self.gp_optimizer.update_experiment(trial_index, metrics, metadata)
        
        # 更新到 LLM
        self.llm_agent.add_experiment_result(parameters, metrics, metadata)
    
    def get_best_parameters(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        获取当前最优参数配置
        
        Returns:
            best_parameters: 最优参数配置
            best_metrics: 最优指标值
        """
        return self.gp_optimizer.get_best_parameters()
    
    def get_optimization_history(self):
        """获取优化历史记录"""
        return self.gp_optimizer.get_optimization_history()


def example_usage():
    """使用示例"""
    from LLINBO_agent import ProblemContext, Parameter, LLMConfig
    import os
    
    # 1. 定义问题背景
    problem_context = ProblemContext(
        problem_description="优化激光切割工艺参数，以提高切割质量和效率",
        industry="制造业 - 激光加工",
        domain_knowledge="激光功率、切割速度和频率对表面粗糙度和切缝宽度有显著影响",
        constraints=["功率不能超过设备上限", "速度必须保证切割质量"],
        optimization_goals=["最小化表面粗糙度", "最小化切缝宽度"]
    )
    
    # 2. 定义 LLM 参数空间
    llm_parameters = [
        Parameter(
            name="power",
            type="range",
            bounds=[1000, 3000],
            value_type="int",
            description="激光功率",
            unit="W"
        ),
        Parameter(
            name="speed",
            type="range",
            bounds=[10.0, 50.0],
            value_type="float",
            description="切割速度",
            unit="mm/s"
        ),
        Parameter(
            name="frequency",
            type="choice",
            values=[500, 1000, 1500, 2000],
            value_type="int",
            description="脉冲频率",
            unit="Hz"
        )
    ]
    
    # 3. 定义 GP 参数空间（Ax 格式）
    gp_search_space = [
        {
            "name": "power",
            "type": "range",
            "bounds": [1000, 3000],
            "value_type": "int"
        },
        {
            "name": "speed",
            "type": "range",
            "bounds": [10, 50],
            "value_type": "float"
        },
        {
            "name": "frequency",
            "type": "choice",
            "values": [500, 1000, 1500, 2000],
            "value_type": "int"
        }
    ]
    
    # 4. 定义优化配置
    optimization_config = {
        "objectives": {
            "roughness": {"minimize": True},
            "kerf_width": {"minimize": True}
        }
    }
    
    # 5. 创建融合优化器
    llm_config = LLMConfig(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"
    )
    
    hybrid_optimizer = HybridOptimizer(
        llm_parameters=llm_parameters,
        gp_search_space=gp_search_space,
        optimization_config=optimization_config,
        problem_context=problem_context,
        llm_config=llm_config,
        acquisition_threshold=0.1,  # 固定阈值（当 use_dynamic_threshold=False 时使用）
        use_dynamic_threshold=True,  # 使用基于方差的动态阈值
        threshold_multiplier=1.0,  # 动态阈值倍数（阈值 = threshold_multiplier * 预测标准差）
        random_seed=42
    )
    
    # 6. 添加先验数据
    prior_experiments = [
        ExperimentResult(
            parameters={"power": 2000, "speed": 30.0, "frequency": 1000},
            metrics={"roughness": 2.5, "kerf_width": 0.15}
        ),
        ExperimentResult(
            parameters={"power": 2500, "speed": 40.0, "frequency": 1500},
            metrics={"roughness": 1.8, "kerf_width": 0.18}
        )
    ]
    hybrid_optimizer.add_prior_experiments(prior_experiments)
    
    # 7. 生成融合推荐
    suggestions = hybrid_optimizer.suggest_parameters(
        num_suggestions=3,
        use_llm=True,
        use_gp=True,
        print_details=True
    )
    
    print("\n📊 最终接受的推荐:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n推荐 {i}:")
        for param_name, param_value in suggestion.items():
            print(f"  {param_name}: {param_value}")


if __name__ == "__main__":
    example_usage()

