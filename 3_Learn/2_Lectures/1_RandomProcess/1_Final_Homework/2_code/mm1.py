import heapq
import os
import numpy as np
from collections import namedtuple
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

plt.rcParams['font.family'] = list({"SimHei", "Heiti TC"} & set(f.name for f in mpl.font_manager.fontManager.ttflist))[0]

# 定义事件类型
Event = namedtuple('Event', ['time', 'type', 'data'])
# type: 'arrival', 'departure', 'rate_change'

class ControlledMM1Queue:
    
    def __init__(self, 
                 arrival_rate: float,
                 service_rate_policy: Callable[[int], float],
                 service_cost_func: Callable[[float], float],
                 queue_cost_func: Callable[[int], float],
                 max_customers: int = None):
        self.arrival_rate = arrival_rate
        self.service_rate_policy = service_rate_policy
        self.service_cost_func = service_cost_func
        self.queue_cost_func = queue_cost_func
        self.max_customers = max_customers
        
        # 系统状态
        self.current_time = 0.0
        self.num_customers = 0
        self.current_service_rate = 0.0
        
        # 事件堆 (优先队列)
        self.event_queue: List[Event] = []
        
        # 统计数据
        self.history = {
            'time': [0.0],
            'num_customers': [0],
            'service_rate': [0.0],
            'total_cost': [0.0]
        }
        self.total_cost = 0.0
        self.last_event_time = 0.0
        
    def reset(self):
        """重置系统状态"""
        self.current_time = 0.0
        self.num_customers = 0
        self.current_service_rate = 0.0
        self.event_queue = []
        self.history = {
            'time': [0.0],
            'num_customers': [0],
            'service_rate': [0.0],
            'total_cost': [0.0]
        }
        self.total_cost = 0.0
        self.last_event_time = 0.0
    
    def _add_event(self, time: float, event_type: str, data=None):
        """添加事件到优先队列"""
        heapq.heappush(self.event_queue, Event(time, event_type, data))
    
    def _exponential_sample(self, rate: float) -> float:
        """从指数分布采样时间间隔"""
        if rate <= 0:
            return float('inf')
        return np.random.exponential(1.0 / rate)
    
    def _schedule_arrival(self):
        """生成下一个到达事件"""
        inter_arrival = self._exponential_sample(self.arrival_rate)
        next_arrival_time = self.current_time + inter_arrival
        self._add_event(next_arrival_time, 'arrival')
    
    def _schedule_departure(self):
        """生成下一个离开事件（如果系统在忙）"""
        if self.num_customers > 0:
            service_rate = self.service_rate_policy(self.num_customers)
            service_time = self._exponential_sample(service_rate)
            next_departure_time = self.current_time + service_time
            self._add_event(next_departure_time, 'departure')
            self.current_service_rate = service_rate
        else:
            self.current_service_rate = 0.0
    
    def _update_cost(self):
        """累计从上事件到现在的代价"""
        dt = self.current_time - self.last_event_time
        if dt > 0:
            # 排队代价
            queue_cost = self.queue_cost_func(self.num_customers) * dt
            
            # 服务代价
            service_cost = self.service_cost_func(self.current_service_rate) * dt
            
            self.total_cost += queue_cost + service_cost
        
        # 记录历史
        self.history['time'].append(self.current_time)
        self.history['num_customers'].append(self.num_customers)
        self.history['service_rate'].append(self.current_service_rate)
        self.history['total_cost'].append(self.total_cost)
        
        self.last_event_time = self.current_time
    
    def _handle_arrival(self):
        """处理到达事件"""
        # 更新代价
        self._update_cost()
        
        # 检查容量限制
        if self.max_customers is None or self.num_customers < self.max_customers:
            self.num_customers += 1
            
            # 如果系统从空变为非空，启动服务
            if self.num_customers == 1:
                self._schedule_departure()
        
        # 安排下一个到达
        self._schedule_arrival()
    
    def _handle_departure(self):
        """处理离开事件"""
        # 更新代价
        self._update_cost()
        
        if self.num_customers > 0:
            self.num_customers -= 1
            
            # 如果仍有顾客，为下一个顾客安排服务
            if self.num_customers > 0:
                self._schedule_departure()
            else:
                self.current_service_rate = 0.0
    
    def run(self, T: float) -> dict:
        """
        运行模拟
        """
        self.reset()
        
        self._schedule_arrival()
        
        # 主事件循环
        while self.event_queue:
            # 获取最近的事件
            event = heapq.heappop(self.event_queue)
            
            # 检查是否超出模拟时间
            if event.time > T:
                break
            
            # 执行事件
            self.current_time = event.time
            
            if event.type == 'arrival':
                self._handle_arrival()
            elif event.type == 'departure':
                self._handle_departure()
            
            print(f"Simulating... {self.current_time/T*100:.2f}", end='\r')
        
        # 最后更新
        self.current_time = T
        self._update_cost()
        
        return self.history
    
    def get_average_cost(self, T: float = None) -> float:
        """计算时间平均代价"""
        if T is None:
            T = self.history['time'][-1]
        return self.total_cost / T
    
    def plot_history(self, figsize=(12, 9), save=False, save_dir="./"):
        """可视化系统演化"""
        fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=300, sharex=True)
        
        # 顾客数量
        axes[0].step(self.history['time'], self.history['num_customers'], 
                     where='post', linewidth=2)
        axes[0].set_ylabel('顾客数量')
        axes[0].set_title('M/M/1 排队系统动态')
        axes[0].grid(True, alpha=0.3)
        
        # 服务速率
        axes[1].step(self.history['time'], self.history['service_rate'], 
                     where='post', linewidth=2, color='orange')
        axes[1].set_ylabel('服务速率 μ')
        axes[1].grid(True, alpha=0.3)
        
        # 累计代价
        axes[2].step(self.history['time'], self.history['total_cost'], 
                     where='post', linewidth=2, color='red')
        axes[2].set_ylabel('累计代价')
        axes[2].set_xlabel('时间')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(save_dir, f"controlled_MM1_L_{self.arrival_rate}.svg"), format="svg")
        else:
            plt.show()
        
        return fig
