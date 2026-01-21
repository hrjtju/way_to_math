import numpy as np
from typing import Callable, List
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

plt.rcParams['font.family'] = list({"SimHei", "Heiti TC"} & set(f.name for f in mpl.font_manager.fontManager.ttflist))[0]


class CTMDPControlledQueue:
    """
    CTMDP 带控制队列求解器
    保证值迭代和策略迭代的数学一致性
    """
    
    def __init__(self, 
                 lambda_rate: float,
                 max_state: int,
                 c_func: Callable[[int], float],
                 q_func: Callable[[float], float],
                 mu_space: List[float],
                 beta: float = 0.1):
        self.lambda_rate = lambda_rate
        self.max_state = max_state
        self.c_func = c_func
        self.q_func = q_func
        self.mu_space = np.array(sorted(mu_space))
        self.beta = beta
        
        # 均匀化速率
        self.V = lambda_rate + np.max(self.mu_space)
        
        if not np.any(np.isclose(self.mu_space, 0.0)):
            raise ValueError("μ_space 必须包含 0")
        
        self._c_vec = np.array([c_func(i) for i in range(max_state + 1)])
        self._q_vec = np.array([q_func(mu) for mu in self.mu_space])
        
    def _bellman_rhs(self, i: int, mu: float, J: np.ndarray) -> float:
        """计算Bellman方程的右侧表达式（统一使用均匀化版本）"""
        if i == 0:
            # 状态0：μ必须为0
            return self._c_vec[i] + (self.V - self.lambda_rate) * J[i] + self.lambda_rate * J[i+1]
        elif i == self.max_state:
            # 状态M：无i+1转移
            return self._c_vec[i] + self.q_func(mu) + mu * J[i-1] + (self.V - mu) * J[i]
        else:
            return (self._c_vec[i] + self.q_func(mu) + 
                    mu * J[i-1] + (self.V - self.lambda_rate - mu) * J[i] + 
                    self.lambda_rate * J[i+1])
        
    def value_iteration(self, tolerance: float = 1e-6, max_iter: int = 20000) -> tuple:
        """标准值迭代（使用均匀化，保证收缩性）"""
        J = np.zeros(self.max_state + 1)
        policy = np.zeros(self.max_state + 1)
        
        pb = tqdm(range(max_iter))
        for iteration in pb:
            J_new = np.zeros_like(J)
            
            for i in range(self.max_state + 1):
                if i == 0:
                    J_new[i] = self._bellman_rhs(i, 0.0, J) / (self.beta + self.V)
                    policy[i] = 0.0
                else:
                    # 评估所有μ
                    rhs_values = np.array([
                        self._bellman_rhs(i, mu, J) for mu in self.mu_space
                    ])
                    best_idx = np.argmin(rhs_values)
                    policy[i] = self.mu_space[best_idx]
                    J_new[i] = rhs_values[best_idx] / (self.beta + self.V)
            
            diff = np.max(np.abs(J_new - J))
            if diff < tolerance:
                print(f"✓ 值迭代收敛于第 {iteration} 次，差值: {diff:.2e}")
                break
            
            pb.set_postfix({"Diff": f"{diff:.3e}"})
            J = J_new
            
            if iteration == max_iter - 1:
                print("⚠ 达到最大迭代次数")
        
        return J, policy
    
    def policy_iteration(self, max_iter: int = 500) -> tuple:
        """
        策略评估使用与值迭代完全相同的均匀化算子
        求解: J = (c + Q̃J)/(β+V) 通过线性系统 (β+V)J - Q̃J = c
        """
        num_states = self.max_state + 1
        policy = np.full(num_states, self.mu_space[0])
        J = np.zeros(num_states)
        
        # print("\n=== 均匀化版本策略迭代 ===")
        # print(f"求解: (β+V)J - Q̃J = c")
        
        for iteration in range(max_iter):
            # 构建: A = (β+V)I - Q̃
            main_diag = np.full(num_states, self.beta + self.V)
            upper_diag = np.full(num_states - 1, -self.lambda_rate)
            lower_diag = np.full(num_states - 1, 0.0)
            b = np.zeros(num_states)

            # 状态0: μ=0
            # (β+V)J0 - (V-λ)J0 - λJ1 = c0
            # => (β+λ)J0 - λJ1 = c0
            b[0] = self._c_vec[0]
            main_diag[0] = self.beta + self.lambda_rate
            # upper_diag[0] = -self.lambda_rate 已设置

            # 状态1..M-1
            for i in range(1, self.max_state):
                mu_i = policy[i]
                lower_diag[i-1] = -mu_i
                # (β+V)Ji - μJi-1 - (V-λ-μ)Ji - λJi+1 = ci + q(μ)
                # => -(μ)Ji-1 + (β+λ+μ)Ji - λJi+1 = ci + q(μ)
                main_diag[i] = self.beta + self.lambda_rate + mu_i
                b[i] = self._c_vec[i] + self.q_func(mu_i)

            # 状态M
            mu_M = policy[self.max_state]
            lower_diag[self.max_state - 1] = -mu_M
            # (β+V)JM - μJM-1 - (V-μ)JM = cM + q(μ)
            # => -(μ)JM-1 + (β+μ)JM = cM + q(μ)
            main_diag[self.max_state] = self.beta + mu_M
            b[self.max_state] = self._c_vec[self.max_state] + self.q_func(mu_M)

            # 求解
            A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
            J_new = spsolve(A, b)
            
            # 验证求解结果是否满足Bellman方程
            if iteration == 0:
                for i in range(min(3, num_states)):
                    if i == 0:
                        lhs = (self.beta + self.lambda_rate) * J_new[i] - self.lambda_rate * J_new[i+1]
                        rhs = b[i]
                    elif i == self.max_state:
                        mu = policy[i]
                        lhs = -mu * J_new[i-1] + (self.beta + mu) * J_new[i]
                        rhs = b[i]
                    else:
                        mu = policy[i]
                        lhs = -mu * J_new[i-1] + (self.beta + self.lambda_rate + mu) * J_new[i] - self.lambda_rate * J_new[i+1]
                        rhs = b[i]

            # 策略改进：使用与值迭代一致的argmin
            new_policy = np.zeros(num_states)
            new_policy[0] = 0.0
            
            for i in range(1, num_states):
                if i == self.max_state:
                    # 边界状态
                    rhs_values = np.array([
                        self._c_vec[i] + self.q_func(mu) + mu * J_new[i-1] + (self.V - mu) * J_new[i]
                        for mu in self.mu_space
                    ])
                else:
                    rhs_values = np.array([
                        self._c_vec[i] + self.q_func(mu) + mu * J_new[i-1] + 
                        (self.V - self.lambda_rate - mu) * J_new[i] + self.lambda_rate * J_new[i+1]
                        for mu in self.mu_space
                    ])
                best_idx = np.argmin(rhs_values)
                new_policy[i] = self.mu_space[best_idx]
            
            # 收敛判断
            policy_diff = np.max(np.abs(new_policy - policy))
            J_diff = np.max(np.abs(J_new - J))
            
            # if iteration % 5 == 0:
            #     print(f"\n迭代 {iteration:2d}: policy_diff={policy_diff:.8f}, J_diff={J_diff:.8f}")
            #     print(f"  J(0..3): {J_new[:3]}")
            
            if policy_diff < 1e-8 and J_diff < 1e-8:
            #     print(f"\n✓ 收敛于第 {iteration} 次迭代")
                residual = self.compute_bellman_residual(J_new, new_policy)
            #     print(f"  Bellman残差: {residual:.2e}")
                break
            
            policy = new_policy
            J = J_new.copy()

        return J, policy
    
    def compute_bellman_residual(self, J: np.ndarray, policy: np.ndarray) -> float:
        """计算Bellman残差（值迭代版本）"""
        residual = 0.0
        
        for i in tqdm(range(self.max_state + 1), desc="验证Bellman残差"):
            rhs = self._bellman_rhs(i, policy[i], J)
            T_J = rhs / (self.beta + self.V)
            residual = max(residual, abs(J[i] - T_J))
        
        return residual
    
    def compare_policies(self, J_vi: np.ndarray, policy_vi: np.ndarray,
                         J_pi: np.ndarray, policy_pi: np.ndarray,
                         num_states: int = 20):
        """详细对比值迭代和策略迭代结果"""
        print("\n" + "="*80)
        print("值迭代 vs 策略迭代 对比")
        print("="*80)
        
        print(f"J差异(max): {np.max(np.abs(J_vi - J_pi)):.10f}")
        print(f"策略差异(max): {np.max(np.abs(policy_vi - policy_pi)):.10f}")
        
        print("\n  状态  |  J_VI(i)  |  J_PI(i)  |  ΔJ_VI |  ΔJ_PI | μ*_VI | μ*_PI | 是否一致")
        print("-"*80)
        
        mismatch_count = 0
        for i in range(min(num_states, self.max_state) + 1):
            delta_J_vi = J_vi[i] - J_vi[i-1] if i > 0 else 0.0
            delta_J_pi = J_pi[i] - J_pi[i-1] if i > 0 else 0.0
            policy_match = "✓" if np.isclose(policy_vi[i], policy_pi[i], atol=1e-6) else "✗"
            if policy_match == "✗":
                mismatch_count += 1
            
            print(f"  {i:3d}   |  {J_vi[i]:8.4f}  |  {J_pi[i]:8.4f}  |  {delta_J_vi:6.2f} |  "
                  f"{delta_J_pi:6.2f} |  {policy_vi[i]:4.1f}  |  {policy_pi[i]:4.1f}  |  {policy_match}")
        
        if mismatch_count == 0:
            print("\n✓ 两种方法策略完全一致！")
        else:
            print(f"\n⚠ 警告：前{num_states}个状态中有{mismatch_count}个状态策略不同")
        
        # 计算残差
        residual_vi = self.compute_bellman_residual(J_vi, policy_vi)
        residual_pi = self.compute_bellman_residual(J_pi, policy_pi)
        print(f"\nBellman残差 - 值迭代: {residual_vi:.2e}, 策略迭代: {residual_pi:.2e}")

def compute_average_cost_steady_state(solver, policy):
    """
    通过求解稳态平衡方程计算精确平均代价
    """
    lambda_rate = solver.lambda_rate
    max_state = solver.max_state
    
    # 构建稳态转移速率矩阵 Q
    Q = np.zeros((max_state+1, max_state+1))
    
    for i in range(max_state+1):
        mu = policy[i]
        
        if i < max_state:
            Q[i, i+1] = lambda_rate  # 到达转移
        
        if i > 0:
            Q[i, i-1] = mu  # 服务转移
        
        # 对角线（离开速率）
        Q[i, i] = -(lambda_rate + mu)
    
    # 求解稳态分布 πQ = 0, Σπ = 1
    # 使用特征向量法
    eigenvals, eigenvecs = np.linalg.eig(Q.T)
    zero_idx = np.argmin(np.abs(eigenvals))
    pi = np.abs(eigenvecs[:, zero_idx].real)
    pi /= np.sum(pi)
    
    # 计算平均代价
    costs = np.array([
        solver.c_func(i) + solver.q_func(policy[i])
        for i in range(max_state+1)
    ])
    
    average_cost = np.dot(pi, costs)
    
    # 验证归一化
    print(f"稳态分布验证: Σπ = {np.sum(pi):.6f}")
    print(f"前10个状态稳态概率: {pi[:10]}")
    
    return average_cost

# 参数配置（建议使用更精细的离散化）
if __name__ == "__main__":
    # LAMBDA = 10.0
    # MAX_STATE = 100  # 增大状态空间
    # BETA = 0.01
    
    def linear(x):
        return x

    def quadratic(x):
        return 0.5 * x ** 2
    
    def exponential(x):
        return np.exp(0.1 * x)
    
    # 使用更密集的μ空间避免边界问题
    mu_space = np.linspace(0, 50, 100)

    ls = []
    
    # for c_func in [linear, quadratic, exponential]:
    #     for q_func in [linear, quadratic, exponential]:
            
    #         solver = CTMDPControlledQueue(
    #             lambda_rate=LAMBDA,
    #             max_state=MAX_STATE,
    #             c_func=c_func,
    #             q_func=q_func,
    #             mu_space=mu_space,
    #             beta=BETA
    #         )
            
    #         J_pi, policy_pi = solver.policy_iteration()
    #         ls.append((c_func.__name__, q_func.__name__, J_pi, policy_pi))
            
    #         from main import ControlledMM1Queue
    
    #         def service_rate_policy(a):
    #             return policy_pi[a]
            
    #         queue = ControlledMM1Queue(
    #             arrival_rate=LAMBDA,
    #             service_rate_policy=service_rate_policy,
    #             service_cost_func=q_func,
    #             queue_cost_func=c_func,
    #             max_customers=MAX_STATE
    #         )
            
    #         queue.run(1e5)
    #         avg_cost = compute_average_cost_steady_state(solver, policy_pi)
    #         print(f"Stationary average cost (steady state): {avg_cost:.4f}")
    #         print(f"Stationary average cost (simulation): {queue.get_average_cost():.4f}")
    
    # plt.figure(figsize=(12, 8), dpi=300)
    
    # for i, (c_name, q_name, J_pi, policy_pi) in enumerate(ls):
    #     plt.subplot(3, 3, i+1)
    #     plt.plot(policy_pi)
    #     plt.title(f"Queue cost {c_name} + Service cost {q_name}")    
    #     plt.grid()
    
    #     if i > 5:
    #         plt.xlabel("State $i$")
        
    #     if i % 3 == 0:
    #         plt.ylabel("Optimal Service Speed $\mu_i$")
    
    # plt.tight_layout()
    # plt.savefig("ctmdp_value_iteration.png")
    
    c_func = linear
    q_func = quadratic
    
    for N in [100, 1000, 2000]:
        for beta in [0.001, 0.01, 0.1]:
            for LAMBDA in [5, 10, 20]:
                solver = CTMDPControlledQueue(
                    lambda_rate=LAMBDA,
                    max_state=N,
                    c_func=c_func,
                    q_func=q_func,
                    mu_space=mu_space,
                    beta=beta
                )
                
                J_pi, policy_pi = solver.policy_iteration()
                ls.append((N, beta, LAMBDA, J_pi, policy_pi))
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 9), dpi=300)
    axs = axs.flatten()
    
    for i, (N, beta, LAMBDA, J_pi, policy_pi) in enumerate(ls):
        print(i // 3, N, beta, LAMBDA, policy_pi.shape)
        axs[i//3].plot(policy_pi, label=f"$\lambda = {LAMBDA}$")
        axs[i//3].set_title(f"N={N}, beta={beta}")    
        axs[i//3].grid()
        axs[i//3].legend()
    
    fig.tight_layout()
    fig.savefig("ctmdp_policy_iteration_1.png")
    
    # print("\n" + "="*80)
    # print("Policy iteration results")
    # print("="*80)
    # solver.compare_policies(J_vi, policy_vi, J_pi, policy_pi)
    
    # # 
    # # np.savez('ctmdp_converged_results.npz', 
    # #          J_vi=J_vi, policy_vi=policy_vi,
    # #          J_pi=J_pi, policy_pi=policy_pi,
    # #          lambda_rate=LAMBDA, beta=BETA, max_state=MAX_STATE)
    
    # avg_cost = compute_average_cost_steady_state(solver, policy_pi)
    # print(f"Stationary average cost: {avg_cost:.4f}")
    
    
