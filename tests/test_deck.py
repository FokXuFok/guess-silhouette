"""随机抽题（Deck）测试：一轮不重复、跳过排到队尾、轮次进度。"""
import random

from app.deck import Deck


def test_no_repeat_within_round():
    d = Deck(["a", "b", "c", "d"], rng=random.Random(0))
    seen = [d.current]
    for _ in range(3):
        seen.append(d.next())
    assert sorted(seen) == ["a", "b", "c", "d"]


def test_new_round_starts_and_avoids_immediate_repeat():
    d = Deck(["a", "b"], rng=random.Random(0))
    d.next()          # 第二轮第一张
    last = d.current
    nxt = d.next()    # 第二轮第二张
    assert {last, nxt} == {"a", "b"}
    third = d.next()  # 第三轮第一张，不应与上一题相同
    assert third != nxt


def test_skip_defers_current_to_end_of_round():
    d = Deck(["a", "b", "c"], rng=random.Random(1))
    first = d.current
    new = d.skip()
    assert new is not None and new != first
    seen = [new, d.next()]
    assert first not in seen
    assert d.next() == first


def test_empty_deck():
    d = Deck([])
    assert d.current is None
    assert d.next() is None
    assert d.skip() is None
    assert d.total == 0


def test_progress_resets_on_new_round():
    d = Deck(["a", "b"], rng=random.Random(0))
    assert d.shown_in_round == 1
    assert d.total == 2
    d.next()
    assert d.shown_in_round == 2
    d.next()
    assert d.shown_in_round == 1


def test_single_item_skip_keeps_current():
    d = Deck(["only"], rng=random.Random(0))
    assert d.skip() == "only"


# ---------- 不重复模式：展示完即止（不给"再来一轮"） ----------

def test_no_restart_deck_exhausts_after_all_shown():
    d = Deck(["a", "b"], rng=random.Random(0), auto_restart=False)
    first = d.current
    second = d.next()
    assert {first, second} == {"a", "b"}      # 两张不重复
    assert d.next() is None                   # 没有更多了
    assert d.exhausted is True
    assert d.current is second                # 最后一题保留，便于界面展示


def test_no_restart_deck_exhausts_on_next_called_after_last():
    d = Deck(["a"], rng=random.Random(0), auto_restart=False)
    assert d.exhausted is False
    assert d.next() is None
    assert d.exhausted is True


def test_no_restart_skip_on_last_image_exhausts():
    d = Deck(["a"], rng=random.Random(0), auto_restart=False)
    assert d.skip() is None
    assert d.exhausted is True


def test_no_restart_skip_defers_within_round():
    d = Deck(["a", "b", "c"], rng=random.Random(2), auto_restart=False)
    first = d.current
    d.skip()
    shown = [d.current, d.next()]
    assert first not in shown
    assert d.next() == first                  # 被跳过的最后才出现
    assert d.next() is None                   # 之后即展示完