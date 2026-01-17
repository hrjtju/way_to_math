import numpy as np
from typing import Callable, List
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

plt.rcParams['font.family'] = list({"SimHei", "Heiti TC"} & set(f.name for f in mpl.font_manager.fontManager.ttflist))[0]


class CTMDPValueIterationSolver:
    """
    CTMDP 标准值迭代求解器（折扣代价问题）
    
    求解: J_β(i) = min_μ E[ ∫_0^∞ e^{-βt} (c(i(t)) + q(μ(t))) dt ]
    """
    
    def __init__(self, 
                 lambda_rate: float,
                 max_state: int,
                 c_func: Callable[[int], float],
                 q_func: Callable[[float], float],
                 mu_space: List[float],
                 beta: float = 0.1):
        """
        参数：
        - lambda_rate: 到达率 λ
        - max_state: 最大状态数（截断）
        - c_func: 队列代价函数 c(i)
        - q_func: 服务速率代价函数 q(μ)
        - mu_space: 可用的服务速率集合（必须包含0）
        - beta: 折扣因子 (β > 0)
        """
        self.lambda_rate = lambda_rate
        self.max_state = max_state
        self.c_func = c_func
        self.q_func = q_func
        self.mu_space = np.array(sorted(mu_space))
        self.beta = beta
        
        # 均匀化速率 V = λ + max(μ)
        self.V = lambda_rate + np.max(self.mu_space)
        
        # 验证 μ=0 在策略空间中
        if not np.any(np.isclose(self.mu_space, 0.0)):
            raise ValueError("μ_space 必须包含 0（关闭服务的选项）")
        
        # 预计算代价向量
        self._c_vec = np.array([c_func(i) for i in range(max_state + 2)])
        self._q_vec = np.array([q_func(mu) for mu in self.mu_space])
        
    def value_iteration(self, 
                       tolerance: float = 1e-6, 
                       max_iter: int = 20000) -> tuple:
        """
        标准值迭代算法求解折扣代价问题
        
        迭代格式：
        J_{k+1}(i) = min_μ [ c(i) + q(μ) + μJ_k(i-1) + (V-λ-μ)J_k(i) + λJ_k(i+1) ] / (β+V)
        
        返回：
        - J: 最优折扣代价函数
        - policy: 最优确定性策略
        """
        # 初始化值函数
        J = np.zeros(self.max_state + 2)  # 多一个状态用于边界
        policy = np.zeros(self.max_state + 1)
        
        pb = tqdm(range(max_iter))
        for iteration in pb:
            J_new = np.zeros_like(J)
            
            # 批量更新所有状态
            for i in range(self.max_state + 1):
                if i == 0:
                    # 状态0: μ 强制为 0
                    J_new[i] = (self._c_vec[i] + 
                                (self.V - self.lambda_rate) * J[i] + 
                                self.lambda_rate * J[i+1]) / (self.beta + self.V)
                else:
                    # 核心更新: min_μ [c(i) + q(μ) + μJ(i-1) + (V-λ-μ)J(i) + λJ(i+1)]
                    
                    policy[i] = self.mu_space[
                        np.argmin(self._q_vec - self.mu_space * (J[i] - J[i-1]))
                    ]
                    
                    # 对每个μ计算右边表达式
                    rhs_values = self._c_vec[i] + self._q_vec + self.mu_space * J[i-1] \
                                + (self.V - self.lambda_rate - self.mu_space) * J[i] \
                                    +self.lambda_rate * J[i+1]
                    
                    # 选择使右边最小的μ
                    J_new[i] = np.min(rhs_values) / (self.beta + self.V)
                    
            
            # 边界条件: J(max_state+1) = J(max_state)
            J_new[self.max_state + 1] = J_new[self.max_state]
            
            # 检查收敛 (使用无穷范数)
            diff = np.max(np.abs(J_new - J))
            if diff < tolerance:
                print(f"✓ 值迭代收敛于第 {iteration} 次迭代，差值: {diff:.2e}")
                break
            
            pb.set_postfix({"Diff": f"{diff:.3e}"})
            
            J = J_new
            
            if iteration == max_iter - 1:
                print("⚠ 达到最大迭代次数，可能未完全收敛")
        
        # 状态0的策略固定为0
        policy[0] = 0.0
        
        return J[:self.max_state+1], policy
    
    def policy_iteration(self, tolerance: float = 1e-6, 
                       max_iter: int = 20000) -> tuple:
        # 1. 初始化策略 (从空间中选一个固定的 mu)
        # policy 始终为 0，因为状态 0 不可控 [4]
        policy = np.full(self.max_state + 1, self.mu_space[-1])
        
        J = np.zeros(self.max_state + 1)

        for iteration in range(max_iter):
            # --- 步骤一：策略评估 (解线性方程组 Ax = B) ---
            # 构造三对角矩阵 A 的数据
            main_diag = np.zeros(self.max_state + 1)
            upper_diag = np.zeros(self.max_state)
            lower_diag = np.zeros(self.max_state)
            B = np.zeros(self.max_state + 1)

            # 状态 0 的方程 [4]
            main_diag[:] = self.beta + self.lambda_rate
            upper_diag[:] = -self.lambda_rate
            B = self._c_vec

            # 状态 1 到 max_state-1 的方程 [4]
            for i in range(1, self.max_state):
                mu_i = policy[i]
                lower_diag[i-1] = -mu_i
                main_diag[i] = self.beta + self.lambda_rate + mu_i
                upper_diag[i] = -self.lambda_rate
                B[i] = self._c_vec[i] + self.q_func(mu_i)

            # 边界状态 max_state (假设 J[N+1] = J[N])
            main_diag[self.max_state] = self.beta + policy[self.max_state] # 简化处理
            lower_diag[-1] = -policy[self.max_state]
            B[self.max_state] = self._c_vec[self.max_state] + self.q_func(policy[self.max_state])

            # 解 A*J = B
            A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
            J_new = spsolve(A, B)

            # --- 步骤二：策略改进 ---
            new_policy = np.zeros_like(policy)
            for i in range(1, self.max_state + 1):
                # 根据 Lec4 [5] 的公式提取最优 mu
                # mu* = argmin [ q(mu) - mu * (J(i) - J(i-1)) ]
                diff_J = J_new[i] - J_new[i-1]
                idx = np.argmin(self._q_vec - self.mu_space * diff_J)
                new_policy[i] = self.mu_space[idx]
            
            # --- 步骤三：终止检查 ---
            if np.array_equal(new_policy, policy):
                print(f"策略迭代在第 {iteration} 次收敛")
                break
            
            policy = new_policy
            J = J_new

        return J, policy
        
    
    def get_greedy_policy(self, J: np.ndarray) -> np.ndarray:
        """
        根据给定的值函数提取贪心策略
        （用于策略迭代或验证）
        """
        policy = np.zeros(self.max_state + 1)
        
        for i in range(1, self.max_state + 1):
            rhs_values = np.zeros_like(self.mu_space)
            
            for idx, mu in enumerate(self.mu_space):
                rhs_values[idx] = (
                    self._c_vec[i] + self._q_vec[idx] +
                    mu * J[i-1] +
                    (self.V - self.lambda_rate - mu) * J[i] +
                    self.lambda_rate * J[i+1]
                )
            
            best_idx = np.argmin(rhs_values)
            policy[i] = self.mu_space[best_idx]
        
        return policy
    
    def compute_bellman_residual(self, J: np.ndarray, policy: np.ndarray) -> float:
        """
        计算 Bellman 残差 max_i |J(i) - T(J)(i)|
        用于验证收敛质量
        """
        residual = 0.0
        
        for i in tqdm(range(self.max_state + 1)):
            if i == 0:
                T_J = (self._c_vec[i] + 
                       (self.V - self.lambda_rate) * J[i] + 
                       self.lambda_rate * J[i+1]) / (self.beta + self.V)
            elif self.max_state:
                mu = policy[i]
                mu_idx = np.where(np.isclose(self.mu_space, mu))[0][0]
                T_J = (
                    self._c_vec[i] + self._q_vec[mu_idx] +
                    mu * J[i-1] +
                    (self.V - mu) * J[i]
                ) / (self.beta + self.V)
            else:
                mu = policy[i]
                mu_idx = np.where(np.isclose(self.mu_space, mu))[0][0]
                T_J = (
                    self._c_vec[i] + self._q_vec[mu_idx] +
                    mu * J[i-1] +
                    (self.V - self.lambda_rate - mu) * J[i] +
                    self.lambda_rate * J[i+1]
                ) / (self.beta + self.V)
            
            residual = max(residual, abs(J[i] - T_J))
        
        return residual
    
    def print_results(self, J: np.ndarray, policy: np.ndarray, num_states: int = 20):
        """格式化输出结果"""
        print("\n" + "="*70)
        print("CTMDP 标准值迭代结果（折扣代价）")
        print("="*70)
        print(f"到达率 λ = {self.lambda_rate:.3f}")
        print(f"折扣因子 β = {self.beta:.3f}")
        print(f"均匀化速率 V = λ + max(μ) = {self.V:.3f}")
        print(f"有效收缩因子 = V / (β+V) = {self.V / (self.beta + self.V):.4f}")
        print(f"状态空间大小 = 0..{self.max_state}")
        
        print("\n  状态  |  折扣代价 J(i)  |  ΔJ = J(i)-J(i-1)  | 最优μ*(i)")
        print("-"*65)
        
        for i in range(min(num_states, self.max_state) + 1):
            delta_J = J[i] - J[i-1] if i > 0 else 0.0
            print(f"  {i:3d}   |   {J[i]:10.4f}   |      {delta_J:8.4f}     |   {policy[i]:6.2f}")
        
        if self.max_state > num_states:
            print("  ...")
        
        # 验证收敛质量
        residual = self.compute_bellman_residual(J, policy)
        print(f"\nBellman 残差: {residual:.2e}")
        if residual < 1e-5:
            print("✓ 通过收敛质量验证！")
        else:
            print("⚠ 残差较大，建议增加迭代次数或检查参数")
    
    def plot_policy_and_value(self, J: np.ndarray, policy: np.ndarray):
        """可视化策略和值函数"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        states = np.arange(self.max_state + 1)
        
        # 最优策略
        axes[0, 0].plot(states, policy, 'bo-', linewidth=2)
        axes[0, 0].set_xlabel('系统顾客数 i')
        axes[0, 0].set_ylabel('最优服务速率 μ*(i)')
        axes[0, 0].set_title('最优控制策略')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 折扣代价函数
        axes[0, 1].plot(states, J, 'r.-', linewidth=2)
        axes[0, 1].set_xlabel('状态 i')
        axes[0, 1].set_ylabel('折扣代价 J(i)')
        axes[0, 1].set_title('最优折扣代价函数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ΔJ
        delta_J = np.diff(J)
        axes[1, 0].plot(states[1:], delta_J, 'g.-', linewidth=2)
        axes[1, 0].set_xlabel('状态 i')
        axes[1, 0].set_ylabel('ΔJ(i) = J(i)-J(i-1)')
        axes[1, 0].set_title('值函数差分')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 服务代价 vs 队列代价
        service_costs = np.array([self.q_func(mu) for mu in policy])
        queue_costs = np.array([self.c_func(i) for i in states])
        axes[1, 1].plot(states, service_costs, 'm.-', label='服务代价 q(μ*(i))')
        axes[1, 1].plot(states, queue_costs, 'c.-', label='队列代价 c(i)')
        axes[1, 1].set_xlabel('状态 i')
        axes[1, 1].set_ylabel('单位时间代价')
        axes[1, 1].set_title('最优策略下的代价分解')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# 使用示例
if __name__ == "__main__":
    # 参数配置
    LAMBDA = 2.0
    MAX_STATE = 1000
    BETA = 0.1  # 折扣因子
    
    # 代价函数
    def c_func(i):
        """队列代价: 线性"""
        return 2.0 * i
    
    def q_func(mu):
        """服务代价: 二次"""
        return 0.5 * mu ** 2
    
    # 服务速率空间
    mu_space = np.linspace(0, 100, 20)  # 0.0 到 10.0
    
    # 创建求解器
    solver = CTMDPValueIterationSolver(
        lambda_rate=LAMBDA,
        max_state=MAX_STATE,
        c_func=c_func,
        q_func=q_func,
        mu_space=mu_space,
        beta=BETA
    )
    
    # 运行值迭代
    print(f"开始标准值迭代（β={BETA}）...")
    # J_optimal, policy_optimal = solver.value_iteration()
    J_optimal, policy_optimal = solver.policy_iteration()
    
    # 显示结果
    solver.print_results(J_optimal, policy_optimal)
    
    # 可视化
    fig = solver.plot_policy_and_value(J_optimal, policy_optimal)
    plt.show()
    
    # 保存结果
    np.savez('discounted_optimal_policy.npz', 
             J=J_optimal, policy=policy_optimal,
             lambda_rate=LAMBDA, beta=BETA, max_state=MAX_STATE)