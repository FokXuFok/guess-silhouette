"""随机抽题：一轮不重复；默认自动重洗，可切换成"展示完即止"。"""
import random


class Deck:
    """题目队列。

    items 可以是任意对象（Entry 等），按身份去重比较。
    auto_restart=True：一轮展示完后自动重新打乱，开新一轮；
    auto_restart=False：所有题目展示完即止，next()/skip() 返回 None 且 exhausted=True。
    """

    def __init__(self, items, rng: random.Random | None = None, auto_restart: bool = True):
        self.items = list(items)
        self.rng = rng or random.Random()
        self.auto_restart = auto_restart
        self.exhausted = False
        self.current = None
        self._queue: list = []
        self._shown = 0
        self._last = None
        if self.items:
            self._start_round()

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def shown_in_round(self) -> int:
        return self._shown

    def _start_round(self) -> None:
        order = list(self.items)
        self.rng.shuffle(order)
        # 新一轮第一题避免与上一轮最后一题重复（题目多于一张时）
        if len(order) > 1 and order[0] is self._last:
            order[0], order[1] = order[1], order[0]
        self._queue = order
        self.current = self._queue.pop(0)
        self._shown = 1

    def next(self):
        if not self.items:
            return None
        self._last = self.current
        if not self._queue:
            if not self.auto_restart:
                self.exhausted = True
                return None
            self._start_round()
            return self.current
        self.current = self._queue.pop(0)
        self._shown += 1
        return self.current

    def skip(self):
        if not self.items or self.current is None:
            return None
        skipped = self.current
        self._last = skipped
        if not self._queue:
            if not self.auto_restart:
                self.exhausted = True
                return None
            if len(self.items) == 1:
                return self.current  # 只有一张图，跳不了
            self._start_round()
            return self.current
        self._queue.append(skipped)  # 跳到本轮队尾
        self.current = self._queue.pop(0)
        self._shown += 1
        return self.current