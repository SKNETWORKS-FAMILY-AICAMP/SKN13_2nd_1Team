# SKN13_2nd_1Team
SKN 13기 2차 단위 프로젝트 레포지토리

-- Discord & git test


## 변수 설명
변수명 | 변수 설명
`#`| 예약 번호
`book_tod` | 현재 예약 시간대 (afternoon, morning, evening, Unknown)
`book_dow` | 현재 예약 요일 (Monday - Sunday)
`book_category` | 현재 예약한 서비스 (STYLE, COLOR, MISC(기타))
`book_staff` | 현재 예약 직원 (JJ, BECKY, Other)
`last_category` | 지난 예약 서비스 (Unknown, STYLE, COLOR, MISC)
`last_staff` | 지난 예약 직원 (Unknown, JJ, Other)
`last_day_services` |  지난 예약 시 받은 서비스의 수 (0 - 3)
`last_receipt_tot` | 지난 예약 때 지불한 금액
`last_dow` | 지난 예약 요일 (Monday - Sunday)
`last_tod` | 지난 예약 시간대 (afternoon, morning, evening, Unknown)
`last_noshow` | 지난 예약 때 노쇼 여부 (1: 노쇼, 0: 방문)
`last_prod_flag` | 지난 예약 시 미용실에서 상품을 구매했는가 (1: 구매, 0: 미구매)
`last_cumrev` | 고객이 지불한 누적 금액
`last_cumbook` | 고객의 누적 예약 횟수
`last_cumstyle` | 고객의 누적 예약 횟수 - STYLE
`last_cumcolor` | 고객의 누적 예약 횟수 - COLOR
`last_cumpord` | 고객의 누적 상품 구매 횟수
`last_cumcancel` | 고객의 누적 예약 취소 횟수
`last_cumnoshow` | 고객의 누적 노쇼 횟수
`noshow` | 현재 예약 노쇼 여부 (1: 노쇼, 0: 방문)
`recency` | 지난 예약 이후 고객이 방문하기까지 간격일
`first_visit` | 현재 예약이 고객의 첫 예약인지 여부 (1: 첫 예약, 0: 예약 이력 있음)
`is_revisit_30days` | 한 달 이내 재방문인지 여부. (1: 재방문, 0: 첫 방문 혹은 한 달 이후 재방문)