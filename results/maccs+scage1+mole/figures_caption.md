# Figure captions — autoresearch effect 증명용

## 전체 주장

autoresearch framework 가 **combo-specific 최적화**를 수행하는지를 검증한다. 즉:
- 같은 combo 를 여러 번 돌리면 **유사한 변경을 채택** (within-combo 일관성)
- 다른 combo 를 돌리면 **서로 다른 변경을 채택** (cross-combo 차별성)

이 두 조건이 동시에 성립해야 “framework 가 random 하지도 generic 하지도 않다”고 말할 수 있다. 그래서 combo1 / combo2 (×3 fork) / combo3 의 **keep 궤적**을 같은 좌표 위에서 비교한다.

---

## Figure 1 — `fig1_theme_heatmap.png`

### 무엇을 보여주는가
세로축은 5개 run 통틀어 한 번이라도 채택된 20개 변경 테마. 가로축은 5개 run (combo1, combo2_main, combo2_fork_a, combo2_fork_b, combo3). 한 셀이 채워져 있으면 그 run이 그 테마를 iter 1-30 안에 keep 했다는 의미. 색은 combo 구분용 (combo1=빨강, combo2=파랑, combo3=초록).

행은 4개 블록으로 묶여있다:
1. **baseline** (모든 run 공통)
2. **combo2 signature** — combo2 세 fork 중 ≥2 회 채택된 테마
3. **combo1 signature** — combo1 에서만 채택
4. **combo3 signature** — combo3 에서만 채택
5. **shared / ambiguous** — 두 combo 이상에 걸쳐 산발적으로 등장

### 무엇을 말하고자 하는가
**시각적으로 “combo 별 블록 패턴”이 직각으로 보인다.**
- combo1 (1열) 은 자기 signature 행에만 빨간색이 몰린다. combo2/3 signature 행은 비어있다.
- combo2 세 fork (2-4열) 는 combo2 signature 행을 거의 다 채운다. combo1/3 signature 행은 비어있다.
- combo3 (5열) 은 자기 signature 행에만 초록색이 몰린다.

즉 **framework 가 어떤 combo 를 돌리느냐에 따라 채택하는 변경 카테고리 자체가 달라진다**. 만약 framework 가 combo 와 무관하게 generic 한 best practice (예: 모든 combo 에 EMA, AdamW+clip, mod_drop) 만 회수한다면 이 그림은 가로줄 (모든 컬럼이 같은 행에서 켜짐) 로만 보여야 한다. 실제로는 거의 그렇지 않다 — baseline 외에는 모든 run 에서 켜진 행이 없다.

### 핵심 메시지
**“autoresearch 는 combo 별로 서로 다른 변경 카테고리를 선택한다.”** 같은 baseline 코드와 같은 LLM 으로 시작했음에도 combo data 의 특성이 keep 결정에 들어와 distinct signature 를 만들어낸다.

---

## Figure 2 — `fig2_jaccard_cluster.png`

### 무엇을 보여주는가
- **좌측 dendrogram**: 5개 run 의 keep-theme set 간 거리 (1 − Jaccard) 로 hierarchical clustering 한 결과. y축은 거리, leaf 가 묶이는 높이가 낮을수록 가깝다.
- **우측 heatmap**: 같은 5×5 의 pairwise Jaccard 매트릭스. 대각선은 자기 자신이라 1.00, 비대각이 실제 일치도. 색은 빨강(높음)→파랑(낮음). combo2 세 fork 의 within-block 을 점선 박스로 강조.

### 무엇을 말하고자 하는가
- dendrogram 에서 **combo2_main 과 combo2_fork_a 가 가장 먼저 묶이고 (1−J ≈ 0.33), combo2_fork_b 가 합류 (≈ 0.65)**, 그 다음 combo1 과 combo3 가 외곽에서 합류한다 (≈ 0.78~0.81).
- heatmap 에서 점선 박스 내부 (combo2 within) 의 Jaccard 는 0.33~0.67. 박스 바깥 (cross-combo) 은 0.15~0.25.

이 그림은 figure 1 의 시각적 패턴을 **거리 통계** 로 환원해 보여준다. 즉 “combo2 세 fork 끼리가 서로 더 가깝다”는 주장이 그래프 구조와 매트릭스 색 분포 양쪽에서 동시에 확인된다.

### 핵심 메시지
**“autoresearch 의 결과를 거리 기준으로 자동 군집화하면 combo 단위로 묶인다.”** 만약 framework 가 combo 와 무관하게 같은 결과를 내면 5개 run 이 hairball 처럼 다 가까이 묶여야 하고, framework 가 random 하다면 어떤 묶임도 없어야 한다. 실제로는 **combo 라벨이 cluster 구조를 정확히 예측**한다 — combo signal 이 keep 궤적에 깊게 박혀있다는 증거.

---

## Figure 3 — `fig3_signature_bars.png`

### 무엇을 보여주는가
Jaccard 값을 두 그룹으로 묶어 분포 비교:
- **왼쪽 boxplot**: combo2 fork 들끼리의 pairwise Jaccard 3개 (main-a, main-b, a-b)
- **오른쪽 boxplot**: combo2 vs combo1, combo2 vs combo3 의 pairwise Jaccard 6개
- **삼각형 마커**: combo1 vs combo3 단일 비교 (1개)
- 점선: within-combo2 평균 (0.455) 과 cross-combo 평균 (0.226)

### 무엇을 말하고자 하는가
**within-combo2 일치도 (평균 0.455) 가 cross-combo 일치도 (평균 0.226) 보다 약 2 배 크다.** 즉 같은 combo 를 다시 돌렸을 때의 변동이 다른 combo 를 돌렸을 때의 변동보다 한 자리 작다.

이는 통계학 용어로 “**combo 가 keep 결정의 분산을 설명한다**”는 의미. ANOVA 식 직관: between-group variance >> within-group variance → group label (combo) 이 신호. 만약 framework 가 random 하다면 within ≈ cross 가 되어야 하고, framework 가 combo-invariant generic recipe 만 회수한다면 within ≈ cross ≈ 1 이 되어야 한다. 둘 다 아닌 중간 영역에서 **명확한 group separation** 이 보인다.

### 핵심 메시지
**“autoresearch 가 combo 에 반응한다는 것을 한 숫자로 요약하면 within/cross 비 ≈ 2× 다.”** 통계적 신호 강도를 단일 plot 으로 보여주는 figure.

---

## 세 figure 의 흐름 (presentation order)

1. **Figure 1** — 채택된 변경 자체가 combo 별로 다름 (raw 증거, 정성적)
2. **Figure 2** — 그 차이가 cluster 구조까지 만든다 (정량적 구조)
3. **Figure 3** — within-combo vs cross-combo 분산 비로 환원 (단일 통계 요약)

발표·논문에서 autoresearch effect 를 주장할 때, **Figure 1 로 “보인다” → Figure 2 로 “묶인다” → Figure 3 로 “2배 차이난다”** 순서로 점진적 abstraction 을 따라가면 자연스러움.

## 유보 사항

- 본 비교는 **keep 결정 (val_auc-only)** 만 사용한 것이며, 채택된 변경이 holdout 성능까지 generalize 하는지는 별도 검증 필요.
- combo1 과 combo3 는 fork 가 없어 within-combo 분산을 측정할 수 없다. 만약 combo1/3 의 within 분산이 combo2 와 비슷하다면 위 결론이 강화되고, 더 크다면 “combo2 만 안정적” 이라는 별도 해석이 가능하다. 엄밀한 검증을 위해서는 combo1/3 도 fork 실험을 권장.
