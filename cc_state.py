"""
cc_state.py — 贾维斯语音交互状态机

管理语音交互的完整生命周期：
    IDLE → 检测到人声 → 录音 → STT/LLM 推理 → TTS 播放 → 回到待机

支持 barge-in（用户打断）、工具调用（飞书等）、超时回退。
每次状态转移记录时间戳，用于性能监控和瓶颈分析。

用法：
    from cc_state import StateMachine, State, Event
    sm = StateMachine()
    sm.transition(Event.SPEECH_START)   # IDLE → LISTENING
    sm.transition(Event.SPEECH_END)     # LISTENING → PROCESSING
    metrics = sm.get_metrics()
"""

import time
import logging
from enum import Enum, auto
from typing import Optional, Callable, Dict, Tuple

logger = logging.getLogger(__name__)


# ── 状态定义 ──

class State(Enum):
    """语音交互状态"""
    IDLE = auto()          # 静默待机，麦克风在听但不录
    LISTENING = auto()     # 检测到人声，开始录音
    PROCESSING = auto()    # STT + LLM 推理中
    SPEAKING = auto()      # TTS 播放中（可被打断）
    INTERRUPTED = auto()   # 刚被打断，等待新语音输入
    TOOL_CALLING = auto()  # 工具调用中（飞书等）


# ── 事件定义 ──

class Event(Enum):
    """触发状态转移的事件"""
    SPEECH_START = auto()  # VAD 检测到人声开始
    SPEECH_END = auto()    # VAD 检测到语音结束（静音超过阈值）
    PROCESS_DONE = auto()  # STT/LLM 处理完成，有回复
    FIRST_AUDIO = auto()   # TTS 首个音频帧就绪
    PLAY_DONE = auto()     # TTS 播放完毕
    BARGE_IN = auto()      # 播放中检测到用户打断
    TOOL_CALL = auto()     # 触发工具调用
    TOOL_DONE = auto()     # 工具调用完成
    TIMEOUT = auto()       # 超时
    ERROR = auto()         # 异常


# ── 转移规则表 ──
# (当前状态, 事件) → 目标状态

TRANSITIONS: Dict[Tuple[State, Event], State] = {
    (State.IDLE, Event.SPEECH_START):        State.LISTENING,
    (State.LISTENING, Event.SPEECH_END):     State.PROCESSING,
    (State.LISTENING, Event.TIMEOUT):        State.IDLE,
    (State.PROCESSING, Event.FIRST_AUDIO):   State.SPEAKING,
    (State.PROCESSING, Event.TOOL_CALL):     State.TOOL_CALLING,
    (State.PROCESSING, Event.ERROR):         State.IDLE,
    (State.SPEAKING, Event.BARGE_IN):        State.INTERRUPTED,
    (State.SPEAKING, Event.PLAY_DONE):       State.IDLE,
    (State.INTERRUPTED, Event.SPEECH_START): State.LISTENING,
    (State.INTERRUPTED, Event.TIMEOUT):      State.IDLE,
    (State.TOOL_CALLING, Event.TOOL_DONE):   State.SPEAKING,
    (State.TOOL_CALLING, Event.ERROR):       State.IDLE,
}


# ── 状态机 ──

class StateMachine:
    """
    语音交互状态机。

    线程安全：transition() 内部无锁，调用方需自行保证单线程驱动。
    """

    def __init__(
        self,
        on_transition: Optional[Callable[[State, Event, State], None]] = None,
    ):
        self.current_state: State = State.IDLE
        self.on_transition = on_transition

        # 性能指标
        self.time_in_state: Dict[State, float] = {s: 0.0 for s in State}
        self.transition_count: int = 0
        self.last_transition_ts: float = time.monotonic()

    def transition(self, event: Event) -> State:
        """
        根据当前状态和事件执行状态转移。

        非法转移不会崩溃，仅打印警告并保持当前状态。

        Returns:
            转移后的状态
        """
        key = (self.current_state, event)
        target = TRANSITIONS.get(key)

        now = time.monotonic()
        elapsed = now - self.last_transition_ts

        if target is None:
            logger.warning(
                "非法转移: %s + %s → ??? (忽略)",
                self.current_state.name, event.name,
            )
            return self.current_state

        # 累计当前状态耗时
        self.time_in_state[self.current_state] += elapsed

        prev = self.current_state
        self.current_state = target
        self.transition_count += 1
        self.last_transition_ts = now

        logger.debug(
            "状态转移: %s + %s → %s (耗时 %.3fs)",
            prev.name, event.name, target.name, elapsed,
        )

        # 回调通知（UI 更新、埋点等）
        if self.on_transition:
            try:
                self.on_transition(prev, event, target)
            except Exception as e:
                logger.error("on_transition 回调异常: %s", e)

        return target

    def get_metrics(self) -> dict:
        """
        返回性能指标字典。

        包含各状态累计耗时、总转移次数、上次转移距今时长。
        """
        now = time.monotonic()
        # 把当前状态的实时耗时也算进去
        live_time = dict(self.time_in_state)
        live_time[self.current_state] += now - self.last_transition_ts

        return {
            "current_state": self.current_state.name,
            "transition_count": self.transition_count,
            "seconds_since_last_transition": round(now - self.last_transition_ts, 3),
            "time_in_state": {
                s.name: round(t, 3) for s, t in live_time.items()
            },
        }

    def reset(self) -> None:
        """重置状态机到初始状态，清空指标。"""
        self.current_state = State.IDLE
        self.time_in_state = {s: 0.0 for s in State}
        self.transition_count = 0
        self.last_transition_ts = time.monotonic()

    def __repr__(self) -> str:
        return f"<StateMachine state={self.current_state.name} transitions={self.transition_count}>"


# ── 测试 ──

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    print("=== 模拟完整对话周期 ===\n")

    transitions_log = []

    def log_transition(prev: State, event: Event, target: State):
        transitions_log.append((prev, event, target))
        print(f"  {prev.name:15s} + {event.name:15s} → {target.name}")

    sm = StateMachine(on_transition=log_transition)
    print(f"初始状态: {sm.current_state.name}\n")

    # 一轮完整对话：待机 → 听 → 处理 → 说 → 待机
    print("── 第一轮：正常对话 ──")
    sm.transition(Event.SPEECH_START)   # IDLE → LISTENING
    sm.transition(Event.SPEECH_END)     # LISTENING → PROCESSING
    sm.transition(Event.FIRST_AUDIO)    # PROCESSING → SPEAKING
    sm.transition(Event.PLAY_DONE)      # SPEAKING → IDLE

    # 带工具调用的对话
    print("\n── 第二轮：工具调用 ──")
    sm.transition(Event.SPEECH_START)   # IDLE → LISTENING
    sm.transition(Event.SPEECH_END)     # LISTENING → PROCESSING
    sm.transition(Event.TOOL_CALL)      # PROCESSING → TOOL_CALLING
    sm.transition(Event.TOOL_DONE)      # TOOL_CALLING → SPEAKING
    sm.transition(Event.PLAY_DONE)      # SPEAKING → IDLE

    # 带 barge-in 的对话
    print("\n── 第三轮：用户打断 ──")
    sm.transition(Event.SPEECH_START)   # IDLE → LISTENING
    sm.transition(Event.SPEECH_END)     # LISTENING → PROCESSING
    sm.transition(Event.FIRST_AUDIO)    # PROCESSING → SPEAKING
    sm.transition(Event.BARGE_IN)       # SPEAKING → INTERRUPTED
    sm.transition(Event.SPEECH_START)   # INTERRUPTED → LISTENING
    sm.transition(Event.SPEECH_END)     # LISTENING → PROCESSING
    sm.transition(Event.FIRST_AUDIO)    # PROCESSING → SPEAKING
    sm.transition(Event.PLAY_DONE)      # SPEAKING → IDLE

    # 测试非法转移
    print("\n── 非法转移（应打印警告） ──")
    sm.transition(Event.PLAY_DONE)      # IDLE + PLAY_DONE → 非法

    # 测试超时
    print("\n── 超时回退 ──")
    sm.transition(Event.SPEECH_START)   # IDLE → LISTENING
    sm.transition(Event.TIMEOUT)        # LISTENING → IDLE

    # 打印性能指标
    print(f"\n=== 性能指标 ===")
    metrics = sm.get_metrics()
    print(f"当前状态: {metrics['current_state']}")
    print(f"总转移次数: {metrics['transition_count']}")
    print(f"各状态累计耗时:")
    for state, t in metrics["time_in_state"].items():
        if t > 0:
            print(f"  {state:15s}: {t:.3f}s")

    print(f"\n{sm}")
