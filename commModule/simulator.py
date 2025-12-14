import heapq
from typing import Callable


class Event:
    def __init__(self, start_delay: float, callback: Callable, *args, **kwargs):
        """
        参数:
            start_delay: 相对于当前时间的开始延迟
            callback: 事件触发时调用的函数
            *args, **kwargs: 传递给回调函数的参数
        """
        self.start_time = None  # 将在调度时设置为绝对时间
        self.start_delay = start_delay
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.completed = False

    def __lt__(self, other):
        """只按开始时间排序"""
        return self.start_time < other.start_time


class TimeSteppedDiscreteEventSimulator:
    """基于固定时间步进的离散事件仿真器"""
    def __init__(self, time_step: float = 0.1):
        self.event_queue = []        # 使用优先队列(堆)管理事件
        self.current_time = 0        # 当前仿真时间
        self.time_step = time_step   # 仿真步长
        self.running = False

    def schedule_event(self, delay: float, callback: Callable, *args, **kwargs):
        """注册一个新事件
        参数:
            delay: 相对于当前时间的延迟时间
            callback: 事件触发时调用的函数
            *args, **kwargs: 传递给回调函数的参数
        """
        event = Event(delay, callback, *args, **kwargs)
        # 设置事件的绝对开始时间
        event.start_time = self.current_time + delay
        heapq.heappush(self.event_queue, event)

    def run(self, end_time: float = None):
        """
        参数:
            end_time: 仿真结束时间，如果为None则运行所有事件
        """
        self.running = True
        processed_events = 0

        while (self.event_queue or processed_events == 0) and (end_time is None or self.current_time <= end_time):
            # 处理当前时间步内的事件
            events_to_process = []

            # 找出所有应该在该时间步处理的事件（开始时间 <= 当前时间）
            while self.event_queue and self.event_queue[0].start_time <= self.current_time:
                event = heapq.heappop(self.event_queue)
                events_to_process.append(event)

            # 处理注册事件
            for event in events_to_process:
                if not event.completed:
                    # 执行事件回调
                    should_complete = event.callback(*(event.args), **(event.kwargs))
                    # 如果回调返回True，则认为事件完成
                    if should_complete is not False:
                        event.completed = True
                        processed_events += 1
                    else:
                        # 事件未完成，重新放回队列
                        heapq.heappush(self.event_queue, event)
                else:
                    processed_events += 1
            # 推进仿真时间
            self.current_time += self.time_step
            # 打印进度
            if int(self.current_time * 10) % 10 == 0:  # 每1秒打印一次
                print(f"时间: {self.current_time:.1f}s, 待处理事件: {len(self.event_queue)}")

        print(f"仿真结束。处理了 {processed_events} 个事件，最终时间: {self.current_time:.2f}")
        self.running = False

    def reset(self):
        """重置仿真器"""
        if self.running:
            raise RuntimeError("不能在运行时重置")
        self.event_queue = []
        self.current_time = 0


def machine_operation(machine_id: int, sim: TimeSteppedDiscreteEventSimulator):
    print(f"时间 {sim.current_time:.2f}: 机器 {machine_id} 开始操作")
    # 模拟操作需要一定时间
    operation_time = 1.5  # 1.5秒操作时间
    if sim.current_time - getattr(machine_operation, 'last_print', 0) >= 0.5:
        print(f"时间 {sim.current_time:.2f}: 机器 {machine_id} 正在运行...")
        machine_operation.last_print = sim.current_time
    # 如果操作未完成，返回False表示需要继续处理
    if sim.current_time - machine_operation.start_time < operation_time:
        return False
    print(f"时间 {sim.current_time:.2f}: 机器 {machine_id} 完成操作")
    # 完成后，安排下一次操作
    next_delay = 0.5  # 0.5秒后再次开始
    sim.schedule_event(next_delay, machine_operation, machine_id, sim)
    return True


# 创建仿真器，时间步长为0.1秒
sim = TimeSteppedDiscreteEventSimulator(time_step=0.1)

# 注册初始事件（2台机器）
sim.schedule_event(0, machine_operation, 1, sim)
sim.schedule_event(0.2, machine_operation, 2, sim)

# 为回调函数添加属性（用于跟踪）
machine_operation.start_time = 0
machine_operation.last_print = 0

# 运行仿真5秒
sim.run(end_time=5.0)
