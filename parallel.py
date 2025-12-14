import atexit
import os
import sys
import time
import traceback
import enum
from functools import partial as bind


class Parallel:

    # ctor：一个 “构造函数”，用来在子进程里创建被代理的对象
    def __init__(self, ctor, strategy):
        # 底层的工作器，负责跨进程执行任务
        self.worker = Worker(bind(self._respond, ctor), strategy, state=True)
        self.callables = {}
    """
    它会先判断这个名字是方法还是属性；
    如果是方法，就返回一个绑定了 PMessage.CALL 的代理函数；
    如果是属性，就直接返回它的值（通过 PMessage.READ）。
    """
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()
            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)
            else:
                return self.worker(PMessage.READ, name)()
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self):
        self.worker.close()
    # 这是运行在子进程里的响应函数
    @staticmethod
    def _respond(ctor, state, message, name, *args, **kwargs):
        state = state or ctor
        if message == PMessage.CALLABLE:
            assert not args and not kwargs, (args, kwargs)
            result = callable(getattr(state, name))
        elif message == PMessage.CALL:
            result = getattr(state, name)(*args, **kwargs)
        elif message == PMessage.READ:
            assert not args and not kwargs, (args, kwargs)
            result = getattr(state, name)
        return state, result


class PMessage(enum.Enum):
    CALLABLE = 2
    CALL = 3
    READ = 4


class Worker:
    # 存放子进程启动前需要执行的初始化函数,共享变量
    initializers = []

    def __init__(self, fn, strategy="thread", state=False):
        if not state:
            fn = lambda s, *args, fn=fn, **kwargs: (s, fn(*args, **kwargs))
        inits = self.initializers
        # 决定是以普通线程还是守护线程启动子进程
        self.impl = {
            "process": bind(ProcessPipeWorker, initializers=inits),
            "daemon": bind(ProcessPipeWorker, initializers=inits, daemon=True),
        }[strategy](fn)
        self.promise = None

    def __call__(self, *args, **kwargs):
        # 提交一个新任务给底层执行单元
        self.promise and self.promise()  # Raise previous exception if any.
        self.promise = self.impl(*args, **kwargs)
        return self.promise

    def wait(self):
        return self.impl.wait()

    def close(self):
        self.impl.close()


class ProcessPipeWorker:
    def __init__(self, fn, initializers=(), daemon=False):
        import multiprocessing
        import cloudpickle
        # 创建多进程上下文
        self._context = multiprocessing.get_context("spawn")
        #  创建双向管道：主进程用self._pipe，子进程用pipe
        self._pipe, pipe = self._context.Pipe()
        # 序列化函数和初始化器（multiprocessing不支持直接传函数，需用cloudpickle）
        fn = cloudpickle.dumps(fn) # fn：子进程要执行的核心函数（如环境的step/reset）
        initializers = cloudpickle.dumps(initializers) # 子进程启动前的初始化操作（如设随机种子）
        # 创建子进程：目标函数是self._loop（子进程的核心循环）
        self._process = self._context.Process(
            target=self._loop, args=(pipe, fn, initializers), daemon=daemon
        ) # 启动子进程
        self._process.start()
        self._nextid = 0 # 每个任务的唯一ID（避免结果混淆）
        self._results = {} # 缓存子进程返回的结果（key=callid，value=结果）
        # 验证子进程是否正常启动（发送OK消息，等待确认
        assert self._submit(Message.OK)()
        # 注册进程退出钩子：主进程退出时自动关闭子进程
        atexit.register(self.close)

    def __call__(self, *args, **kwargs):
        # 对外接口：主进程调用时，提交RUN任务（如执行env.step）
        return self._submit(Message.RUN, (args, kwargs))

    def wait(self):
        pass

    def close(self):
        try:
            # 1. 发送STOP消息，通知子进程退出
            self._pipe.send((Message.STOP, self._nextid, None))
            self._pipe.close()
        except (AttributeError, IOError):
            pass  # The connection was already closed.
        try:
            # 2. 等待子进程退出（最多等0.1秒）
            self._process.join(0.1)
            # 如果子进程还没退出（exitcode是None），强制杀死
            if self._process.exitcode is None:
                try:
                    os.kill(self._process.pid, 9)
                    time.sleep(0.1)
                except Exception:
                    pass
        except (AttributeError, AssertionError):
            pass

    def _submit(self, message, payload=None):
        callid = self._nextid
        self._nextid += 1
        self._pipe.send((message, callid, payload))
        return Future(self._receive, callid)

    def _receive(self, callid):
        # 循环等待，直到拿到当前callid对应的结果
        while callid not in self._results:
            try:
                # 从管道接收子进程的消息（阻塞，直到有消息）
                message, callid, payload = self._pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker.")
            if message == Message.ERROR:
                raise Exception(payload)
            # 确保接收的是结果消息（避免其他类型消息干扰）
            assert message == Message.RESULT, message
            self._results[callid] = payload
        # 从缓存中取出结果并删除（避免内存泄漏）
        return self._results.pop(callid)

    @staticmethod
    def _loop(pipe, function, initializers):
        try:
            callid = None
            state = None
            import cloudpickle
            # 反序列化：把主进程传的函数和初始化器还原
            initializers = cloudpickle.loads(initializers)
            function = cloudpickle.loads(function) # function是主进程传的核心逻辑（如Parallel._respond）
            [fn() for fn in initializers]
            # 子进程核心循环：持续接收并处理主进程的任务
            while True:
                # 每0.1秒检查一次管道（避免死循环，同时响应键盘中断）
                if not pipe.poll(0.1):
                    continue  # Wake up for keyboard interrupts.
                # 接收主进程的任务
                message, callid, payload = pipe.recv()
                if message == Message.OK:
                    # 响应主进程的启动验证：返回True表示子进程正常
                    pipe.send((Message.RESULT, callid, True))
                elif message == Message.STOP:
                    # 接收停止消息，退出循环（子进程终止）
                    return
                elif message == Message.RUN:
                    args, kwargs = payload
                    state, result = function(state, *args, **kwargs)
                    # 把结果发回主进程
                    pipe.send((Message.RESULT, callid, result))
                else:
                    raise KeyError(f"Invalid message: {message}")
        except (EOFError, KeyboardInterrupt):
            return
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((Message.ERROR, callid, stacktrace))
            return
        finally:
            try:
                pipe.close()
            except Exception:
                pass


class Message(enum.Enum):
    OK = 1
    RUN = 2
    RESULT = 3
    STOP = 4
    ERROR = 5


class Future:
    def __init__(self, receive, callid):
        self._receive = receive
        self._callid = callid
        self._result = None
        self._complete = False

    def __call__(self):
        if not self._complete:
            self._result = self._receive(self._callid)
            self._complete = True
        return self._result

# 单个环境的包装类，提供与并行环境相同的接口
class Damy:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return lambda: self._env.step(action)

    def reset(self):
        return lambda: self._env.reset()
    
    def set_next_obs(self, obs):
        return lambda: self._env.set_next_obs(obs)
