"""Read-only explanations of reconciliation failures; never invent fills."""
from collections import OrderedDict


def reconciliation_report(target):
    groups = OrderedDict()
    for issue in target.get('issues', ()):
        text = str(issue)
        if '보유량' in text or '수량' in text or '배분' in text:
            category = '수량·배분 확인'
            action = '보유종목을 다시 조회해 체결 내역을 갱신하세요. 수동·외부 매매가 있었다면 해당 주문과 모델별 잔여 수량을 대조하세요.'
        elif '체결가' in text or '원가' in text:
            category = '체결 가격 확인'
            action = '주문번호로 체결 내역을 다시 확인하세요. 모의 추정가격은 적용 조건을 충족할 때만 사용하며 확정 손익으로 계산하지 않습니다.'
        elif '출처' in text or '원매수' in text:
            category = '매수 출처 확인'
            action = '아래 매수 기록과 원본 모델 신호를 확인하세요. 현재 선택한 모델을 과거 매수 모델로 지정하지 않습니다.'
        else:
            category = '주문 근거 확인'
            action = '주문·체결 내역에서 접수 여부와 누적 체결량을 확인한 뒤 다시 조회하세요. 확인되지 않은 주문을 재전송하지 마세요.'
        groups.setdefault((category, action), []).append(text)
    lines = []
    if 'expected_quantity' in target:
        lines.append(f"장부 보유 {target['expected_quantity']}주 / 증권사 보유 {target.get('broker_quantity', '미확인')}주\n")
    for (category, action), issues in groups.items():
        lines.extend([category, *issues, '조치: ' + action, ''])
    for lot in target.get('lots', ()):
        lines.append(f"모델: {lot.get('model_title', '미확인')} | 주문번호: {lot.get('buy_order_number') or '미확인'} | 매수 기록: {lot.get('buy_rule_id', lot.get('lot_id', '미확인'))} | 잔여: {lot.get('quantity_remaining', '미확인')}주")
    return '\n'.join(lines) or '장부 불일치가 보고되지 않았습니다.'
