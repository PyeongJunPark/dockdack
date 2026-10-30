"""Concise model lifecycle labels, independent of Qt and order execution."""


def model_runtime_state(*, selected, research, ready, orders_enabled):
    if not selected:
        return '미선택'
    if research:
        return '선택 · 연구 전용 · 주문 없음'
    if not ready:
        return '선택 · 데이터·모델 준비 대기'
    return '판단 준비 · 자동주문 ON' if orders_enabled else '판단 준비 · 자동주문 OFF'
