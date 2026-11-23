/**
 * 틴코스 AI (TeenCos AI) - 메인 스크립트 & 기계학습 엔진
 * 지도학습 k-NN 추천기, NLP 리뷰 감성분석기, CEI 가성비 모델, 다차원 비교 스튜디오
 * 화해(Hwahae) 실데이터 연계 및 색조 호수(Color Options) 큐레이션 엔진 탑재
 */

// 전역 상태 관리
const AppState = {
    currentCategory: 'all',
    selectedProductsForComparison: new Set(),
    userPreferences: {
        category: 'all',
        maxPrice: 30000,
        cleanOnly: false,
        skinType: 'oily',
        skinTroubles: ['acne'],
        personalColor: 'all',
        selectedColorOptions: {}, // { [productId]: optionNo }
        kValue: 3,
        weights: {
            priceWeight: 0.35,
            ingredientWeight: 0.25,
            sentimentWeight: 0.20,
            matchWeight: 0.20
        }
    },
    activeTab: 'recommender', // 'recommender', 'comparator', 'ailab', 'report'
    dataset: [...COSMETICS_DATASET]
};

// --------------------------------------------------------------------------
// 1. 머신러닝 엔진 (ML Engine)
// --------------------------------------------------------------------------

class MLEngine {
    /**
     * NLP 감성 분석기 (Lexicon-based Sentiment Analyzer)
     * 화해 실사용자 리뷰 텍스트를 분석하여 긍정 확률(0~100%)을 도출합니다.
     */
    static analyzeSentiment(reviews) {
        if (!reviews || reviews.length === 0) return { score: 85, positiveRate: 85, topKeywords: ['순함', '가성비'] };

        let totalPos = 0;
        let totalNeg = 0;
        let matchedKeywords = [];

        reviews.forEach(review => {
            // 긍정 사전 매칭
            for (const [word, weight] of Object.entries(SENTIMENT_LEXICON.positive)) {
                if (review.includes(word)) {
                    totalPos += weight;
                    if (!matchedKeywords.includes(word)) matchedKeywords.push(word);
                }
            }
            // 부정 사전 매칭
            for (const [word, weight] of Object.entries(SENTIMENT_LEXICON.negative)) {
                if (review.includes(word)) {
                    totalNeg += Math.abs(weight);
                }
            }
        });

        const totalScore = totalPos + totalNeg;
        const positiveRate = totalScore === 0 ? 88 : Math.round((totalPos / (totalScore + 0.1)) * 100);
        const finalScore = Math.min(100, Math.max(50, positiveRate));

        return {
            score: finalScore,
            positiveRate: finalScore,
            matchedKeywords: matchedKeywords.slice(0, 4)
        };
    }

    /**
     * 용량 대비 단위 가격 계산 (원/ml 또는 원/g)
     */
    static calculateUnitPrice(product) {
        if (!product.capacity || product.capacity <= 0) return product.price;
        return Math.round(product.price / product.capacity);
    }

    /**
     * 성분 안전도 점수 계산 (0~100점)
     * 화해 20가지 주의성분 및 알레르기 유발성분, EWG 그린등급 비율 반영
     */
    static calculateIngredientSafetyScore(product) {
        const cautionPenalty = (product.cautionIngredients || 0) * 20;
        const allergyPenalty = (product.allergyCaution || 0) * 10;
        const ewgScore = product.ewgGreenRate || 90;
        return Math.max(40, Math.min(100, ewgScore - cautionPenalty - allergyPenalty));
    }

    /**
     * 피부 타입 및 고민 일치도 계산 (Jaccard Index 변형, 0~1.0)
     */
    static calculateSkinMatch(userType, userTroubles, product) {
        let score = 0;
        if (product.skinTypes.includes('all') || product.skinTypes.includes(userType)) {
            score += 0.5;
        }

        if (userTroubles && userTroubles.length > 0) {
            const matches = userTroubles.filter(t => product.troubles.includes(t));
            score += (matches.length / userTroubles.length) * 0.5;
        } else {
            score += 0.5;
        }

        return Math.min(1.0, score);
    }

    /**
     * 퍼스널 컬러 일치도 (0~1.0)
     * 선택된 색상 호수의 톤을 우선하여 계산
     */
    static calculateColorMatch(userTone, effectiveTone) {
        if (userTone === 'all' || effectiveTone === 'all') return 1.0;
        if (userTone === effectiveTone) return 1.0;
        if (effectiveTone === 'neutral') return 0.85;
        return 0.35; // 톤이 상이할 경우
    }

    /**
     * 최적 가성비 지수 (CEI, Cost-Effectiveness Index)
     */
    static calculateCEI(product, userPrefs) {
        const unitPrice = this.calculateUnitPrice(product);
        const safetyScore = this.calculateIngredientSafetyScore(product);
        const sentiment = this.analyzeSentiment(product.sampleReviews);
        const skinMatch = this.calculateSkinMatch(userPrefs.skinType, userPrefs.skinTroubles, product);

        // 품질 점수 (0~100)
        const qualityScore = (safetyScore * 0.35) + (sentiment.score * 0.35) + ((skinMatch * 100) * 0.3);

        // 가격 점수 (낮을수록 고득점, 다이소급 극강 가성비에 추가 가중)
        let priceFactor = 1.0;
        if (unitPrice <= 30) priceFactor = 1.40;       // 다이소 초저가
        else if (unitPrice <= 80) priceFactor = 1.25;
        else if (unitPrice <= 180) priceFactor = 1.0;
        else if (unitPrice <= 300) priceFactor = 0.85;
        else priceFactor = 0.70;

        // 화해 평점 보너스 가중
        const hwahaeBonus = (product.hwahaeRating ? product.hwahaeRating - 4.0 : 0.5) * 5;

        // 최종 CEI (최대 99점)
        let cei = Math.round((qualityScore * 0.58) * priceFactor + (product.rating * 6) + hwahaeBonus);
        cei = Math.min(99, Math.max(55, cei));

        // 가성비 등급 부여
        let grade = 'A';
        let gradeBadge = '가성비 우수';
        if (cei >= 93) { grade = 'SS'; gradeBadge = '🏆 갓성비 원탑'; }
        else if (cei >= 85) { grade = 'S'; gradeBadge = '⭐ 가성비 꿀템'; }
        else if (cei >= 75) { grade = 'A'; gradeBadge = '👍 만족템'; }
        else { grade = 'B'; gradeBadge = '일반'; }

        return {
            ceiScore: cei,
            grade: grade,
            gradeBadge: gradeBadge,
            unitPrice: unitPrice,
            safetyScore: safetyScore,
            sentimentScore: sentiment.score,
            positiveRate: sentiment.positiveRate,
            matchedKeywords: sentiment.matchedKeywords,
            qualityScore: Math.round(qualityScore)
        };
    }

    /**
     * 지도학습 k-NN (k-Nearest Neighbors) 기반 최적 화장품 추천
     */
    static predictKNN(userPrefs, dataset) {
        const { category, maxPrice, cleanOnly, skinType, skinTroubles, personalColor, selectedColorOptions, kValue, weights } = userPrefs;

        // 1단계: 카테고리 및 클린 성분 필터링
        let filtered = dataset;
        if (category && category !== 'all') {
            filtered = filtered.filter(p => p.category === category);
        }
        if (cleanOnly) {
            filtered = filtered.filter(p => p.cautionIngredients === 0);
        }

        // 2단계: 각 제품의 다차원 특징 벡터 추출 및 거리 계산
        const scoredItems = filtered.map(product => {
            const metrics = this.calculateCEI(product, userPrefs);
            const skinMatch = this.calculateSkinMatch(skinType, skinTroubles, product);

            // 해당 제품의 활성 컬러 옵션 결정
            let activeColorOpt = null;
            let effectiveTone = product.personalColor;

            if (product.colorOptions && product.colorOptions.length > 0) {
                const userSelectedNo = selectedColorOptions[product.id];
                if (userSelectedNo) {
                    activeColorOpt = product.colorOptions.find(o => o.no === userSelectedNo) || product.colorOptions[0];
                } else if (personalColor !== 'all') {
                    // 사용자 퍼스널컬러에 가장 알맞은 옵션을 자동 스마트 셀렉트
                    activeColorOpt = product.colorOptions.find(o => o.tone === personalColor) || product.colorOptions[0];
                } else {
                    activeColorOpt = product.colorOptions[0];
                }
                effectiveTone = activeColorOpt.tone;
            }

            const colorMatch = this.calculateColorMatch(personalColor, effectiveTone);

            // [특성 정규화 (Min-Max)]
            // f1: 가격 적합도 (희망 예산 이하일수록 1.0, 예산 초과 시 부드러운 감점)
            let fPrice = 1.0;
            if (maxPrice >= 50000) {
                // 무제한 예산일 경우 용량당 단가 기준으로만 평가
                fPrice = Math.max(0.6, 1.0 - (metrics.unitPrice / 500) * 0.4);
            } else if (product.price <= maxPrice) {
                fPrice = 1.0 - (product.price / 60000) * 0.4; // 쌀수록 1.0에 가까움
            } else {
                fPrice = Math.max(0, 1.0 - ((product.price - maxPrice) / 15000));
            }

            // f2: 성분 안전도 (0~1.0)
            const fSafety = metrics.safetyScore / 100;

            // f3: 리뷰 감성 점수 (0~1.0)
            const fSentiment = metrics.sentimentScore / 100;

            // f4: 피부 및 퍼스널컬러 종합 매칭 (0~1.0)
            const fMatch = (skinMatch * 0.65) + (colorMatch * 0.35);

            // 가중 유클리드 거리 (Weighted Euclidean Distance)
            const w = weights;
            const distSq = (
                w.priceWeight * Math.pow(1.0 - fPrice, 2) +
                w.ingredientWeight * Math.pow(1.0 - fSafety, 2) +
                w.sentimentWeight * Math.pow(1.0 - fSentiment, 2) +
                w.matchWeight * Math.pow(1.0 - fMatch, 2)
            );
            const distance = Math.sqrt(distSq);

            // 유사도 퍼센티지 (Distance -> Similarity 변환)
            const similarity = Math.round((1 / (1 + distance)) * 100);

            // XAI 설명 생성
            const xaiReasons = [];
            if (product.hwahaeRanking) xaiReasons.push(`🏆 ${product.hwahaeRanking}`);
            if (product.price <= 5000) xaiReasons.push(`단돈 ${product.price.toLocaleString()}원 다이소 갓성비`);
            else if (metrics.unitPrice <= 80) xaiReasons.push(`1${product.unit}당 ${metrics.unitPrice}원으로 동급 최강 가성비`);

            if (metrics.safetyScore >= 95) xaiReasons.push(`화해 20가지 주의성분 0개, EWG 그린등급 ${product.ewgGreenRate}%`);
            if (skinMatch >= 0.8) xaiReasons.push(`${ML_CONFIG.skinTypeMap[skinType].split(' ')[0]} 피부 및 고민 일치`);
            if (activeColorOpt) {
                xaiReasons.push(`추천 컬러: ${activeColorOpt.name} (${activeColorOpt.toneName})`);
            } else if (colorMatch === 1.0 && personalColor !== 'all') {
                xaiReasons.push(`${ML_CONFIG.personalColorMap[personalColor].split(' ')[0]} 찰떡 톤`);
            }
            if (metrics.positiveRate >= 90) xaiReasons.push(`화해 10대 구매자 긍정 리뷰 ${metrics.positiveRate}%`);

            return {
                product,
                metrics,
                activeColorOpt,
                effectiveTone,
                distance: parseFloat(distance.toFixed(4)),
                similarity,
                xaiReasons,
                vector: {
                    priceScore: Math.round(fPrice * 100),
                    safetyScore: Math.round(fSafety * 100),
                    sentimentScore: Math.round(fSentiment * 100),
                    matchScore: Math.round(fMatch * 100),
                    overallCEI: metrics.ceiScore
                }
            };
        });

        // 3단계: 유사도 순(거리 오름차순)으로 정렬
        scoredItems.sort((a, b) => b.similarity - a.similarity || b.metrics.ceiScore - a.metrics.ceiScore);

        return {
            recommendations: scoredItems,
            kNeighbors: scoredItems.slice(0, kValue),
            totalCount: scoredItems.length
        };
    }
}

// --------------------------------------------------------------------------
// 2. UI 렌더링 및 인터랙션 핸들러
// --------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function initApp() {
    renderCategoryNav();
    bindFilterEvents();
    bindTabNavigation();
    bindLabEvents();
    runRecommendation();
}

/**
 * 카테고리 필터 탭 렌더링
 */
function renderCategoryNav() {
    const navContainer = document.getElementById('categoryTabs');
    if (!navContainer) return;

    navContainer.innerHTML = Object.values(CATEGORIES).map(cat => `
        <button class="cat-pill ${AppState.currentCategory === cat.id ? 'active' : ''}" data-cat="${cat.id}">
            <span>${cat.icon}</span> ${cat.name}
        </button>
    `).join('');

    navContainer.querySelectorAll('.cat-pill').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const catId = e.currentTarget.dataset.cat;
            AppState.currentCategory = catId;
            AppState.userPreferences.category = catId;
            document.getElementById('inputCategory').value = catId;
            renderCategoryNav();
            runRecommendation();
        });
    });
}

/**
 * 탭 전환 헬퍼 함수
 */
function switchTab(tab) {
    AppState.activeTab = tab;
    document.querySelectorAll('.nav-tab-btn').forEach(btn => {
        if (btn.dataset.tab === tab) btn.classList.add('active');
        else btn.classList.remove('active');
    });

    document.querySelectorAll('.tab-content-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    const targetPanel = document.getElementById(`panel-${tab}`);
    if (targetPanel) targetPanel.classList.add('active');

    if (tab === 'ailab') {
        updateAILabView();
    } else if (tab === 'comparator') {
        renderComparisonStudio();
    }
}

/**
 * 탭 네비게이션 바인딩
 */
function bindTabNavigation() {
    const tabBtns = document.querySelectorAll('.nav-tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            switchTab(btn.dataset.tab);
        });
    });
}

/**
 * 필터 및 슬라이더 이벤트 바인딩
 */
function bindFilterEvents() {
    const priceSlider = document.getElementById('inputPrice');
    const priceDisplay = document.getElementById('priceDisplay');
    const skinTypeSelect = document.getElementById('inputSkinType');
    const personalColorSelect = document.getElementById('inputColorTone');
    const categorySelect = document.getElementById('inputCategory');
    const troubleCheckboxes = document.querySelectorAll('input[name="skinTrouble"]');
    const prioritySelect = document.getElementById('inputPriority');
    const cleanOnlyCheckbox = document.getElementById('filterCleanOnly');

    if (priceSlider && priceDisplay) {
        priceSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            priceDisplay.textContent = val >= 50000 ? '50,000원+ (제한 없음)' : `최대 ${val.toLocaleString()}원 이하`;
            AppState.userPreferences.maxPrice = val;
            runRecommendation();
        });
    }

    if (cleanOnlyCheckbox) {
        cleanOnlyCheckbox.addEventListener('change', (e) => {
            AppState.userPreferences.cleanOnly = e.target.checked;
            runRecommendation();
        });
    }

    if (skinTypeSelect) {
        skinTypeSelect.addEventListener('change', (e) => {
            AppState.userPreferences.skinType = e.target.value;
            runRecommendation();
        });
    }

    if (personalColorSelect) {
        personalColorSelect.addEventListener('change', (e) => {
            AppState.userPreferences.personalColor = e.target.value;
            runRecommendation();
        });
    }

    if (categorySelect) {
        categorySelect.addEventListener('change', (e) => {
            AppState.userPreferences.category = e.target.value;
            AppState.currentCategory = e.target.value;
            renderCategoryNav();
            runRecommendation();
        });
    }

    troubleCheckboxes.forEach(cb => {
        cb.addEventListener('change', () => {
            const checked = Array.from(document.querySelectorAll('input[name="skinTrouble"]:checked')).map(c => c.value);
            AppState.userPreferences.skinTroubles = checked;
            runRecommendation();
        });
    });

    if (prioritySelect) {
        prioritySelect.addEventListener('change', (e) => {
            const p = e.target.value;
            if (p === 'price') {
                AppState.userPreferences.weights = { priceWeight: 0.50, ingredientWeight: 0.20, sentimentWeight: 0.15, matchWeight: 0.15 };
            } else if (p === 'ingredient') {
                AppState.userPreferences.weights = { priceWeight: 0.20, ingredientWeight: 0.50, sentimentWeight: 0.15, matchWeight: 0.15 };
            } else if (p === 'review') {
                AppState.userPreferences.weights = { priceWeight: 0.20, ingredientWeight: 0.20, sentimentWeight: 0.45, matchWeight: 0.15 };
            } else {
                AppState.userPreferences.weights = { priceWeight: 0.35, ingredientWeight: 0.25, sentimentWeight: 0.20, matchWeight: 0.20 };
            }
            runRecommendation();
        });
    }
}

/**
 * 색상 옵션(호수) 선택 함수
 */
function selectColorOption(productId, optionNo, event) {
    if (event) event.stopPropagation();
    AppState.userPreferences.selectedColorOptions[productId] = optionNo;
    runRecommendation();
}

/**
 * 인공지능 추천 추론 실행 및 카드 렌더링
 */
function runRecommendation() {
    const result = MLEngine.predictKNN(AppState.userPreferences, AppState.dataset);
    renderProductCards(result.recommendations);
    renderTopPickHighlight(result.recommendations[0]);
    updateComparisonBar();
    updateAILabView();
}

/**
 * 1순위 최적 가성비 추천 제품 하이라이트 배너 렌더링
 */
function renderTopPickHighlight(topItem) {
    const highlightEl = document.getElementById('topPickHighlight');
    if (!highlightEl || !topItem) return;

    const { product, metrics, activeColorOpt, similarity, xaiReasons } = topItem;

    // 색상 옵션 스와치 HTML 생성
    let colorOptionsHTML = '';
    if (product.colorOptions && product.colorOptions.length > 0) {
        colorOptionsHTML = `
            <div class="color-options-box">
                <span class="color-box-title">🎨 색상 옵션 (호수 선택):</span>
                <div class="color-swatches-row">
                    ${product.colorOptions.map(opt => `
                        <button type="button" class="swatch-btn ${activeColorOpt && activeColorOpt.no === opt.no ? 'active' : ''}" 
                                style="background: ${opt.hex};" 
                                title="${opt.name} (${opt.toneName})"
                                onclick="selectColorOption('${product.id}', '${opt.no}', event)">
                            <span class="swatch-check">✓</span>
                        </button>
                    `).join('')}
                </div>
                ${activeColorOpt ? `
                    <div class="active-color-desc">
                        선택: <strong>${activeColorOpt.name}</strong> <span class="tone-badge ${activeColorOpt.tone}">${activeColorOpt.toneName}</span>
                        <p class="color-sub-desc">${activeColorOpt.desc}</p>
                    </div>
                ` : ''}
            </div>
        `;
    }

    highlightEl.innerHTML = `
        <div class="top-pick-card">
            <div class="top-badge-ribbon">
                <span class="ribbon-icon">👑</span> AI 판정 1위 최적 가성비 추천템
            </div>
            <div class="top-pick-content">
                <div class="top-pick-visual">
                    <div class="top-product-image-container">
                        <img src="${product.imageUrl}" alt="${product.name}" class="top-product-img" loading="lazy"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                        <div class="top-avatar-fallback" style="display:none;">
                            ${getProductCategoryIcon(product.category)}
                        </div>
                    </div>
                    <div class="top-similarity-pill">
                        AI 매칭 적합도 <strong>${similarity}%</strong>
                    </div>
                </div>
                <div class="top-pick-info">
                    <div class="brand-row-badges">
                        <span class="top-pick-brand">${product.brand} · ${product.categoryName}</span>
                        ${product.hwahaeRanking ? `<span class="hwahae-pill">🏆 ${product.hwahaeRanking}</span>` : ''}
                    </div>
                    <h3 class="top-pick-title">${product.name}</h3>
                    <div class="top-pick-pricing">
                        <span class="top-price">${product.price.toLocaleString()}원</span>
                        <span class="top-unit-price">(${product.capacity}${product.unit} · 1${product.unit}당 ${metrics.unitPrice}원)</span>
                        <span class="cei-badge ${metrics.grade}">${metrics.gradeBadge}</span>
                    </div>

                    ${colorOptionsHTML}

                    <div class="xai-explanation-box">
                        <div class="xai-title">🤖 인공지능 추천 이유 (Explainable AI) :</div>
                        <ul class="xai-list">
                            ${xaiReasons.map(r => `<li><i class="check-icon">✓</i> ${r}</li>`).join('')}
                        </ul>
                    </div>
                    <div class="top-pick-actions">
                        <button class="btn btn-compare-toggle ${AppState.selectedProductsForComparison.has(product.id) ? 'active' : ''}" 
                                onclick="toggleProductComparison('${product.id}')">
                            ${AppState.selectedProductsForComparison.has(product.id) ? '✓ 비교함에 담김' : '+ 다른 화장품과 1:1 비교하기'}
                        </button>
                        <button class="btn btn-detail" onclick="openProductDetailModal('${product.id}')">
                            🔍 화해 실리뷰·성분 상세분석
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 화장품 카드 리스트 렌더링
 */
function renderProductCards(recommendations) {
    const gridContainer = document.getElementById('productsGrid');
    const countBadge = document.getElementById('resultCountBadge');
    if (!gridContainer) return;

    if (countBadge) countBadge.textContent = `${recommendations.length}개 검색됨`;

    if (recommendations.length === 0) {
        gridContainer.innerHTML = `
            <div class="no-result-box">
                <p>😥 조건에 맞는 가성비 화장품이 없습니다.</p>
                <p>예산 슬라이더를 올리거나 필터를 초기화해 보세요!</p>
            </div>
        `;
        return;
    }

    gridContainer.innerHTML = recommendations.map(({ product, metrics, activeColorOpt, similarity, xaiReasons }, index) => {
        const isCompared = AppState.selectedProductsForComparison.has(product.id);
        const rank = index + 1;

        // 색상 옵션 스와치
        let cardColorHTML = '';
        if (product.colorOptions && product.colorOptions.length > 0) {
            cardColorHTML = `
                <div class="card-color-options">
                    <div class="swatches-mini-row">
                        ${product.colorOptions.map(opt => `
                            <button type="button" class="swatch-mini-btn ${activeColorOpt && activeColorOpt.no === opt.no ? 'active' : ''}" 
                                    style="background: ${opt.hex};" 
                                    title="${opt.name} (${opt.toneName})"
                                    onclick="selectColorOption('${product.id}', '${opt.no}', event)">
                            </button>
                        `).join('')}
                    </div>
                    ${activeColorOpt ? `
                        <span class="active-swatch-name">${activeColorOpt.name} <small>(${activeColorOpt.toneName})</small></span>
                    ` : ''}
                </div>
            `;
        }

        return `
            <div class="product-card ${rank <= 3 ? 'rank-top3' : ''}" data-id="${product.id}">
                <div class="card-header-bar">
                    <span class="rank-tag ${rank === 1 ? 'rank-1' : ''}">${rank}위</span>
                    <span class="grade-tag grade-${metrics.grade}">${metrics.gradeBadge}</span>
                    <button class="btn-check-compare ${isCompared ? 'checked' : ''}" 
                            title="비교함에 추가" 
                            onclick="toggleProductComparison('${product.id}')">
                        ${isCompared ? '✓ 담김' : '+ 비교'}
                    </button>
                </div>
                
                <div class="card-body">
                    <!-- 제품 실물 이미지 영역 -->
                    <div class="product-card-visual">
                        <div class="img-wrapper">
                            <img src="${product.imageUrl}" alt="${product.name}" class="product-thumb-img" loading="lazy"
                                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                            <div class="card-avatar-fallback" style="display:none;">
                                ${getProductCategoryIcon(product.category)}
                            </div>
                        </div>
                        <div class="match-mini-badge">${similarity}% 일치</div>
                    </div>

                    <div class="product-brand-cat">
                        <span>${product.brand} · ${product.categoryName}</span>
                    </div>

                    ${product.hwahaeRanking ? `<div class="hwahae-ranking-tag">🏆 ${product.hwahaeRanking}</div>` : ''}

                    <h4 class="product-name" title="${product.name}">${product.name}</h4>

                    <div class="price-row">
                        <span class="price-val">${product.price.toLocaleString()}원</span>
                        <span class="unit-price-val">1${product.unit}당 ${metrics.unitPrice}원</span>
                    </div>

                    <!-- 색상 옵션 스와치 영역 -->
                    ${cardColorHTML}

                    <div class="spec-tags-wrap">
                        <span class="spec-pill ${product.cautionIngredients === 0 ? 'safe' : 'caution'}">
                            ${product.cautionIngredients === 0 ? '🌱 화해 주의성분 0개' : `⚠️ 주의성분 ${product.cautionIngredients}개`}
                        </span>
                        <span class="spec-pill tone-pill">
                            ${formatTone(activeColorOpt ? activeColorOpt.tone : product.personalColor)}
                        </span>
                    </div>

                    <!-- AI 특성 게이지 바 -->
                    <div class="mini-metrics">
                        <div class="metric-line">
                            <span class="metric-label">화해리뷰 긍정</span>
                            <div class="metric-progress">
                                <div class="bar-fill" style="width: ${metrics.positiveRate}%"></div>
                            </div>
                            <span class="metric-num">${metrics.positiveRate}%</span>
                        </div>
                        <div class="metric-line">
                            <span class="metric-label">가성비 지수(CEI)</span>
                            <div class="metric-progress">
                                <div class="bar-fill cei-bar" style="width: ${metrics.ceiScore}%"></div>
                            </div>
                            <span class="metric-num">${metrics.ceiScore}점</span>
                        </div>
                    </div>

                    <!-- 10대 한줄 후기 키워드 -->
                    <div class="review-keyword-chips">
                        ${metrics.matchedKeywords.map(k => `<span class="k-chip">#${k}</span>`).join('')}
                    </div>
                </div>

                <div class="card-footer">
                    <button class="btn-view-spec" onclick="openProductDetailModal('${product.id}')">
                        🔍 화해 성분·리뷰 분석보기
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * 카테고리별 아이콘 매핑
 */
function getProductCategoryIcon(cat) {
    return CATEGORIES[cat]?.icon || '✨';
}

function formatTone(tone) {
    if (tone === 'warm') return '🍊 웜톤';
    if (tone === 'cool') return '🍇 쿨톤';
    if (tone === 'neutral') return '🫧 뉴트럴';
    return '💧 모든 톤';
}

// --------------------------------------------------------------------------
// 3. 비교함 & 화장품 비교 스튜디오 (Comparison Studio)
// --------------------------------------------------------------------------

function toggleProductComparison(productId) {
    if (AppState.selectedProductsForComparison.has(productId)) {
        AppState.selectedProductsForComparison.delete(productId);
    } else {
        if (AppState.selectedProductsForComparison.size >= 4) {
            alert('화장품 비교는 최대 4개까지 동시에 가능합니다!');
            return;
        }
        AppState.selectedProductsForComparison.add(productId);
    }
    runRecommendation();
}

function updateComparisonBar() {
    const bar = document.getElementById('floatingCompareBar');
    const countSpan = document.getElementById('compareCount');
    const chipsContainer = document.getElementById('compareSelectedChips');
    if (!bar) return;

    const count = AppState.selectedProductsForComparison.size;
    if (count > 0) {
        bar.classList.add('visible');
        if (countSpan) countSpan.textContent = count;

        const selectedList = AppState.dataset.filter(p => AppState.selectedProductsForComparison.has(p.id));
        if (chipsContainer) {
            chipsContainer.innerHTML = selectedList.map(p => `
                <div class="chip-item">
                    <img src="${p.imageUrl}" class="chip-thumb" alt="" onerror="this.style.display='none';">
                    <span>${p.name.length > 10 ? p.name.substring(0, 10) + '...' : p.name}</span>
                    <button onclick="toggleProductComparison('${p.id}')">×</button>
                </div>
            `).join('');
        }
    } else {
        bar.classList.remove('visible');
    }
}

function clearAllComparison() {
    AppState.selectedProductsForComparison.clear();
    runRecommendation();
}

function openComparisonModal() {
    if (AppState.selectedProductsForComparison.size < 2) {
        alert('비교를 위해 화장품을 2개 이상 선택해 주세요! (각 제품 카드의 "+ 비교" 버튼 클릭)');
        return;
    }
    switchTab('comparator');
}

/**
 * 비교 스튜디오 렌더링 (레이더 차트 + 1:1 대조 테이블)
 */
function renderComparisonStudio() {
    const container = document.getElementById('comparatorContainer');
    if (!container) return;

    const selectedIds = Array.from(AppState.selectedProductsForComparison);
    if (selectedIds.length < 2) {
        container.innerHTML = `
            <div class="empty-studio-notice">
                <div class="notice-icon">⚖️</div>
                <h3>비교할 화장품이 2개 이상 필요합니다</h3>
                <p>추천 리스트에서 비교하고 싶은 화장품 카드의 <strong>[+ 비교]</strong> 버튼을 2~4개 눌러주세요!</p>
                <button class="btn btn-primary" onclick="switchTab('recommender')">
                    화장품 둘러보러 가기
                </button>
            </div>
        `;
        return;
    }

    const items = selectedIds.map(id => {
        const prod = AppState.dataset.find(p => p.id === id);
        const metrics = MLEngine.calculateCEI(prod, AppState.userPreferences);
        const skinMatch = MLEngine.calculateSkinMatch(AppState.userPreferences.skinType, AppState.userPreferences.skinTroubles, prod);
        const userSelectedNo = AppState.userPreferences.selectedColorOptions[prod.id];
        const activeColorOpt = (prod.colorOptions && prod.colorOptions.length > 0) 
            ? (prod.colorOptions.find(o => o.no === userSelectedNo) || prod.colorOptions[0]) 
            : null;
        return { prod, metrics, skinMatch, activeColorOpt };
    });

    const lowestUnitPriceItem = [...items].sort((a, b) => a.metrics.unitPrice - b.metrics.unitPrice)[0];
    const highestCEIItem = [...items].sort((a, b) => b.metrics.ceiScore - a.metrics.ceiScore)[0];

    container.innerHTML = `
        <div class="comparison-studio-wrapper">
            <div class="studio-header">
                <h2>📊 화장품 다차원 1:1 비교 분석실</h2>
                <p>성분, 단가, 화해 리뷰, 피부 적합도를 5개 축 방사형 차트와 상세 스펙표로 한눈에 대조합니다.</p>
            </div>

            <!-- 레이더 차트 영역 -->
            <div class="radar-chart-section">
                <div class="chart-canvas-wrap">
                    <canvas id="comparisonRadarCanvas" width="480" height="360"></canvas>
                </div>
                <div class="chart-legend-wrap" id="chartLegendWrap"></div>
            </div>

            <!-- AI 1:1 비교 총평 -->
            <div class="ai-comparison-verdict">
                <h4>🤖 인공지능 비교 총평 (Verdict)</h4>
                <div class="verdict-grid">
                    <div class="verdict-box best-price">
                        <strong>💰 최강 단가 1위:</strong>
                        <p>${lowestUnitPriceItem.prod.name} (1${lowestUnitPriceItem.prod.unit}당 ${lowestUnitPriceItem.metrics.unitPrice}원! 학생 용돈 절약에 가장 유리)</p>
                    </div>
                    <div class="verdict-box best-cei">
                        <strong>🏆 종합 가성비 1위:</strong>
                        <p>${highestCEIItem.prod.name} (가성비 지수 ${highestCEIItem.metrics.ceiScore}점, 성분과 화해 만족도 밸런스 최우수)</p>
                    </div>
                </div>
            </div>

            <!-- 1:1 스펙 대조 테이블 -->
            <div class="table-responsive">
                <table class="spec-compare-table">
                    <thead>
                        <tr>
                            <th>비교 항목</th>
                            ${items.map(it => `
                                <th>
                                    <div class="compare-table-thumb">
                                        <img src="${it.prod.imageUrl}" alt="" class="table-img" onerror="this.style.display='none';">
                                    </div>
                                    <div class="th-prod-name">${it.prod.name}</div>
                                    <div class="th-prod-brand">${it.prod.brand}</div>
                                </th>
                            `).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="item-title">소비자가격</td>
                            ${items.map(it => `<td class="price-highlight">${it.prod.price.toLocaleString()}원</td>`).join('')}
                        </tr>
                        <tr>
                            <td class="item-title">용량 및 단위 단가</td>
                            ${items.map(it => `
                                <td>
                                    <strong>1${it.prod.unit}당 ${it.metrics.unitPrice}원</strong><br>
                                    <small>(${it.prod.capacity}${it.prod.unit})</small>
                                </td>
                            `).join('')}
                        </tr>
                        <tr>
                            <td class="item-title">가성비 등급 (CEI)</td>
                            ${items.map(it => `
                                <td>
                                    <span class="cei-badge ${it.metrics.grade}">${it.metrics.gradeBadge}</span><br>
                                    <small>${it.metrics.ceiScore}점 / 100점</small>
                                </td>
                            `).join('')}
                        </tr>
                        <tr>
                            <td class="item-title">화해 랭킹 & 평점</td>
                            ${items.map(it => `
                                <td>
                                    <strong>${it.prod.hwahaeRanking || '화해 랭킹 상위'}</strong><br>
                                    <small>★ ${it.prod.hwahaeRating || it.prod.rating} / 리뷰 ${it.prod.reviewCount.toLocaleString()}건</small>
                                </td>
                            `).join('')}
                        </tr>
                        <tr>
                            <td class="item-title">20가지 주의성분</td>
                            ${items.map(it => `
                                <td>
                                    <span class="badge ${it.prod.cautionIngredients === 0 ? 'badge-safe' : 'badge-caution'}">
                                        ${it.prod.cautionIngredients === 0 ? '0개 (완전 안심)' : `${it.prod.cautionIngredients}개 주의`}
                                    </span>
                                </td>
                            `).join('')}
                        </tr>
                        <tr>
                            <td class="item-title">식약처 알레르기 주의</td>
                            ${items.map(it => `<td>${it.prod.allergyCaution === 0 ? '0개 (안심)' : `${it.prod.allergyCaution}개`}</td>`).join('')}
                        </tr>
                        <tr>
                            <td class="item-title">선택 색상 / 톤</td>
                            ${items.map(it => `
                                <td>
                                    ${it.activeColorOpt ? `
                                        <div style="display:inline-flex; align-items:center; gap:4px;">
                                            <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:${it.activeColorOpt.hex}; border:1px solid #cbd5e1;"></span>
                                            <strong>${it.activeColorOpt.name}</strong>
                                        </div><br>
                                        <small>${it.activeColorOpt.toneName}</small>
                                    ` : `${formatTone(it.prod.personalColor)}<br><small>${it.prod.colorTone}</small>`}
                                </td>
                            `).join('')}
                        </tr>
                        <tr>
                            <td class="item-title">제형 및 마무리감</td>
                            ${items.map(it => `<td>${it.prod.finish}</td>`).join('')}
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;

    drawRadarChart(items);
}

/**
 * HTML5 Canvas 5각 레이더 차트 렌더러
 */
function drawRadarChart(items) {
    const canvas = document.getElementById('comparisonRadarCanvas');
    const legendWrap = document.getElementById('chartLegendWrap');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2 + 10;
    const radius = 120;

    const axes = [
        '단가 경쟁력(저렴도)',
        '성분 안전도',
        '화해리뷰 만족도',
        '피부 고민 적합도',
        '가성비 종합지수'
    ];
    const totalAxes = axes.length;

    ctx.clearRect(0, 0, width, height);

    // 1. 방사형 배경 격자 그리기
    const levels = 4;
    ctx.strokeStyle = '#e2e8f0';
    ctx.fillStyle = '#64748b';
    ctx.font = '12px "Pretendard", sans-serif';

    for (let lvl = 1; lvl <= levels; lvl++) {
        const r = (radius / levels) * lvl;
        ctx.beginPath();
        for (let i = 0; i < totalAxes; i++) {
            const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
            const x = centerX + r * Math.cos(angle);
            const y = centerY + r * Math.sin(angle);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
    }

    // 축 선 및 라벨
    for (let i = 0; i < totalAxes; i++) {
        const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);

        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = '#cbd5e1';
        ctx.stroke();

        const textX = centerX + (radius + 28) * Math.cos(angle);
        const textY = centerY + (radius + 18) * Math.sin(angle);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#334155';
        ctx.fillText(axes[i], textX, textY);
    }

    // 2. 다각형 데이터 그리기
    const colors = [
        { stroke: '#ec4899', fill: 'rgba(236, 72, 153, 0.25)' },
        { stroke: '#3b82f6', fill: 'rgba(59, 130, 246, 0.25)' },
        { stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.25)' },
        { stroke: '#f59e0b', fill: 'rgba(245, 158, 11, 0.25)' }
    ];

    let legendHTML = '';

    items.forEach((item, idx) => {
        const color = colors[idx % colors.length];
        legendHTML += `
            <div class="legend-chip">
                <span class="legend-color-dot" style="background: ${color.stroke}"></span>
                <span class="legend-name">${item.prod.name}</span>
            </div>
        `;

        const priceComp = Math.max(30, Math.min(100, Math.round(100 - (item.metrics.unitPrice / 4))));
        const safety = item.metrics.safetyScore;
        const review = item.metrics.sentimentScore;
        const skinMatch = Math.round(item.skinMatch * 100);
        const cei = item.metrics.ceiScore;

        const values = [priceComp, safety, review, skinMatch, cei];

        ctx.beginPath();
        values.forEach((val, i) => {
            const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
            const r = (val / 100) * radius;
            const x = centerX + r * Math.cos(angle);
            const y = centerY + r * Math.sin(angle);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = color.fill;
        ctx.fill();
        ctx.strokeStyle = color.stroke;
        ctx.lineWidth = 2.5;
        ctx.stroke();

        values.forEach((val, i) => {
            const angle = (Math.PI * 2 / totalAxes) * i - Math.PI / 2;
            const r = (val / 100) * radius;
            const x = centerX + r * Math.cos(angle);
            const y = centerY + r * Math.sin(angle);
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = color.stroke;
            ctx.fill();
        });
    });

    if (legendWrap) legendWrap.innerHTML = legendHTML;
}

// --------------------------------------------------------------------------
// 4. 모달창 (제품 상세 스펙 & 화해 리뷰 모달)
// --------------------------------------------------------------------------

function openProductDetailModal(productId) {
    const product = AppState.dataset.find(p => p.id === productId);
    if (!product) return;

    const metrics = MLEngine.calculateCEI(product, AppState.userPreferences);
    const modalEl = document.getElementById('productDetailModal');
    if (!modalEl) return;

    const userSelectedNo = AppState.userPreferences.selectedColorOptions[product.id];
    const activeColorOpt = (product.colorOptions && product.colorOptions.length > 0)
        ? (product.colorOptions.find(o => o.no === userSelectedNo) || product.colorOptions[0])
        : null;

    let modalColorHTML = '';
    if (product.colorOptions && product.colorOptions.length > 0) {
        modalColorHTML = `
            <div class="modal-section">
                <h4>🎨 색상 호수 옵션 (클릭하여 변경)</h4>
                <div class="modal-color-options-grid">
                    ${product.colorOptions.map(opt => `
                        <div class="modal-color-card ${activeColorOpt && activeColorOpt.no === opt.no ? 'selected' : ''}" 
                             onclick="selectColorOption('${product.id}', '${opt.no}'); openProductDetailModal('${product.id}');">
                            <span class="m-color-circle" style="background:${opt.hex}"></span>
                            <div class="m-color-text">
                                <strong>${opt.name}</strong>
                                <small>${opt.toneName}</small>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    const modalContent = document.getElementById('modalContentBody');
    modalContent.innerHTML = `
        <div class="modal-product-header">
            <div class="modal-product-header-top">
                <img src="${product.imageUrl}" alt="" class="modal-thumb-img" onerror="this.style.display='none';">
                <div class="modal-header-text">
                    <div class="modal-cat">${product.brand} · ${product.categoryName}</div>
                    <h2>${product.name}</h2>
                    ${product.hwahaeRanking ? `<span class="hwahae-pill">🏆 ${product.hwahaeRanking} (★ ${product.hwahaeRating || product.rating})</span>` : ''}
                    <div class="modal-price-bar">
                        <span class="m-price">${product.price.toLocaleString()}원</span>
                        <span class="m-unit">용량: ${product.capacity}${product.unit} (1${product.unit}당 ${metrics.unitPrice}원)</span>
                        <span class="cei-badge ${metrics.grade}">${metrics.gradeBadge}</span>
                    </div>
                </div>
            </div>
        </div>

        ${modalColorHTML}

        <div class="modal-section">
            <h4>🌿 화해(Hwahae) 기준 전성분 안전도 분석</h4>
            <div class="modal-safety-grid">
                <div class="safety-stat">
                    <span>20가지 주의성분</span>
                    <strong style="color: ${product.cautionIngredients === 0 ? '#059669' : '#dc2626'}">
                        ${product.cautionIngredients}개
                    </strong>
                </div>
                <div class="safety-stat">
                    <span>알레르기 주의성분</span>
                    <strong>${product.allergyCaution || 0}개</strong>
                </div>
                <div class="safety-stat">
                    <span>EWG 그린비율</span>
                    <strong>${product.ewgGreenRate}%</strong>
                </div>
            </div>
            <div class="main-ingredients-list">
                <strong>핵심 유효 성분:</strong>
                <div class="ing-tags">
                    ${product.mainIngredients.map(ing => {
                        const dict = INGREDIENT_DICT[ing];
                        return `<span class="ing-tag" title="${dict ? dict.role : '피부 케어'}">${ing} ${dict ? `(${dict.role})` : ''}</span>`;
                    }).join('')}
                </div>
            </div>
        </div>

        <div class="modal-section">
            <h4>💬 화해 실구매자 10대·20대 솔직 후기 (NLP 감성 분석)</h4>
            <div class="sentiment-summary-box">
                <div class="sent-score">
                    감성 긍정율 <strong>${metrics.positiveRate}%</strong>
                </div>
                <div class="sent-keywords">
                    핵심 긍정 키워드: ${metrics.matchedKeywords.map(k => `<code>#${k}</code>`).join(' ')}
                </div>
            </div>
            <div class="modal-reviews-list">
                ${product.sampleReviews.map(r => `
                    <div class="review-bubble">
                        <span class="quote-mark">“</span>
                        <p>${r}</p>
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="modal-actions">
            <button class="btn btn-primary" onclick="toggleProductComparison('${product.id}'); closeProductDetailModal();">
                ${AppState.selectedProductsForComparison.has(product.id) ? '비교함에서 제거' : '+ 비교함에 담기'}
            </button>
            <button class="btn btn-secondary" onclick="closeProductDetailModal()">닫기</button>
        </div>
    `;

    modalEl.classList.add('active');
}

function closeProductDetailModal() {
    const modalEl = document.getElementById('productDetailModal');
    if (modalEl) modalEl.classList.remove('active');
}

// --------------------------------------------------------------------------
// 5. 머신러닝 연구실 (AI Lab) 로직
// --------------------------------------------------------------------------

function bindLabEvents() {
    const kSlider = document.getElementById('labKSlider');
    const kDisplay = document.getElementById('labKDisplay');
    const wPrice = document.getElementById('weightPrice');
    const wIng = document.getElementById('weightIng');
    const wSent = document.getElementById('weightSent');
    const wMatch = document.getElementById('weightMatch');

    if (kSlider) {
        kSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            AppState.userPreferences.kValue = val;
            if (kDisplay) kDisplay.textContent = val;
            runRecommendation();
        });
    }

    const weightInputs = [wPrice, wIng, wSent, wMatch];
    weightInputs.forEach(inp => {
        if (!inp) return;
        inp.addEventListener('input', () => {
            const p = parseFloat(wPrice.value) || 0;
            const i = parseFloat(wIng.value) || 0;
            const s = parseFloat(wSent.value) || 0;
            const m = parseFloat(wMatch.value) || 0;
            const sum = p + i + s + m;

            AppState.userPreferences.weights = {
                priceWeight: p / sum,
                ingredientWeight: i / sum,
                sentimentWeight: s / sum,
                matchWeight: m / sum
            };

            document.getElementById('wPriceVal').textContent = Math.round((p / sum) * 100) + '%';
            document.getElementById('wIngVal').textContent = Math.round((i / sum) * 100) + '%';
            document.getElementById('wSentVal').textContent = Math.round((s / sum) * 100) + '%';
            document.getElementById('wMatchVal').textContent = Math.round((m / sum) * 100) + '%';

            runRecommendation();
        });
    });
}

function updateAILabView() {
    const result = MLEngine.predictKNN(AppState.userPreferences, AppState.dataset);
    const neighborsList = document.getElementById('labNeighborsList');
    if (!neighborsList) return;

    neighborsList.innerHTML = result.kNeighbors.map((item, idx) => `
        <div class="neighbor-item">
            <div class="n-rank">#${idx + 1}</div>
            <img src="${item.product.imageUrl}" class="n-thumb" alt="" onerror="this.style.display='none';">
            <div class="n-info">
                <strong>${item.product.name}</strong>
                <small>${item.product.brand} · ${item.product.price.toLocaleString()}원</small>
            </div>
            <div class="n-metrics">
                <span>유클리드 거리: <code>${item.distance}</code></span>
                <span>유사도: <strong>${item.similarity}%</strong></span>
            </div>
        </div>
    `).join('');
}

// --------------------------------------------------------------------------
// 6. 보고서 인쇄 및 유틸리티
// --------------------------------------------------------------------------

function printReport() {
    window.print();
}
