package main

import (
	"crypto/rand"
	"fmt"
	"log"
	"math"
	"math/big"
	"sort"
	"time"
)

// 目标总 RTP
const totalRtp = 0.97

var model = map[string]float64{
	"normal": 0.975, // 97.5%
	"free":   0.012, // 1.2%
	"bonus":  0.013, // 1.3%
}

var modelRtp = map[string]float64{
	"normal": 0.6,  // 普通
	"free":   20.0, // Free
}

// 倍数集合
var multipliers = map[string][]float64{
	"normal": {0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 10, 20},
	"free":   {5, 6, 7, 8, 9, 10, 11, 12, 14, 20, 23, 25, 27, 30, 35, 37, 40, 45, 50, 55, 60, 100, 200},
	"bonus":  {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 50, 200},
}

const volatility = 0.3

func init() { calcBonusRTP() }

// ----------------- 核心函数 -----------------

func calcBonusRTP() float64 {
	knownContribution := 0.0
	for mode, prob := range model {
		if mode == "bonus" {
			continue
		}
		if rtp, ok := modelRtp[mode]; ok {
			knownContribution += prob * rtp
		}
	}
	bonusProb := model["bonus"]
	bonusRTP := (totalRtp - knownContribution) / bonusProb
	if bonusRTP < 0 {
		log.Fatalf("Bonus RTP 被反推出负数 (%.4f)，输入不合法", bonusRTP)
	}
	modelRtp["bonus"] = bonusRTP
	return bonusRTP
}

// 生成分布（RTP 恒定，波动性可调）
func generateWeights(targetRTP float64, mults []float64, volatility float64) []float64 {
	n := len(mults)
	probs := make([]float64, n)

	minM := mults[0]
	maxM := mults[n-1]
	if targetRTP < minM || targetRTP > maxM {
		log.Fatalf("目标RTP %.4f 不在范围 [%.2f, %.2f]", targetRTP, minM, maxM)
	}

	low, high := -100.0, 100.0
	const eps = 1e-9

	for iter := 0; iter < 200; iter++ {
		mid := (low + high) / 2
		expVals := make([]float64, n)
		sumExp := 0.0
		for i, m := range mults {
			// ★ exp 控制均值，pow 控制波动
			expVals[i] = math.Exp(-mid*m) * math.Pow(m+1, volatility)
			sumExp += expVals[i]
		}
		for i := range expVals {
			probs[i] = expVals[i] / sumExp
		}

		expVal := 0.0
		for i, m := range mults {
			expVal += probs[i] * m
		}

		if math.Abs(expVal-targetRTP) < eps {
			return probs
		}
		if expVal > targetRTP {
			low = mid
		} else {
			high = mid
		}
	}
	return probs
}

// ----------------- 选取器优化 -----------------

// 构建累计分布函数 (CDF)
func buildCDF(probs []float64) []float64 {
	cdf := make([]float64, len(probs))
	sum := 0.0
	for i, p := range probs {
		sum += p
		cdf[i] = sum
	}
	return cdf
}

// 安全随机数 [0,1)
func secureRandFloat() float64 {
	n, _ := rand.Int(rand.Reader, big.NewInt(1<<53))
	return float64(n.Int64()) / (1 << 53)
}

// 优化版 pick (二分查找 O(log n))
func pickMultiplier(mults []float64, cdf []float64) float64 {
	r := secureRandFloat()
	idx := sort.Search(len(cdf), func(i int) bool {
		return cdf[i] >= r
	})
	return mults[idx]
}

func calcVariance(mults []float64, probs []float64, mean float64) float64 {
	variance := 0.0
	for i, m := range mults {
		variance += probs[i] * (m - mean) * (m - mean)
	}
	return variance
}

// ----------------- 主流程 -----------------

func main() {
	start := time.Now()

	// Step1: 生成倍数分布
	modeProbs := make(map[string][]float64)
	modeCDFs := make(map[string][]float64)
	for mode, rtp := range modelRtp {
		modeProbs[mode] = generateWeights(rtp, multipliers[mode], volatility)
		modeCDFs[mode] = buildCDF(modeProbs[mode]) // 构建CDF
		fmt.Printf("模式=%s, RTP=%.2f\n", mode, rtp)
		for i, m := range multipliers[mode] {
			fmt.Printf("  倍数=%.2f, 概率=%.6f\n", m, modeProbs[mode][i])
		}
	}

	// Step2: 计算总体 σ
	mu := 0.0
	for mode, prob := range model {
		mu += prob * modelRtp[mode]
	}
	sigma2 := 0.0
	for mode, prob := range model {
		rtp := modelRtp[mode]
		variance := calcVariance(multipliers[mode], modeProbs[mode], rtp)
		sigma2 += prob * (variance + (rtp-mu)*(rtp-mu))
	}
	sigma := math.Sqrt(sigma2)
	fmt.Printf("\n理论总RTP=%.4f, 标准差σ=%.4f (volatility=%.2f)\n", mu, sigma, volatility)

	// Step3: 模拟
	totalSpins := 200000
	totalBet, totalReturn := 0.0, 0.0
	for i := 0; i < totalSpins; i++ {
		// 选模式
		r := secureRandFloat()
		sum := 0.0
		var chosen string
		for mode, prob := range model {
			sum += prob
			if r <= sum {
				chosen = mode
				break
			}
		}
		// 选倍数 (用二分查找)
		mult := pickMultiplier(multipliers[chosen], modeCDFs[chosen])
		totalBet += 1.0
		totalReturn += mult
	}
	fmt.Printf(
		"\n模拟结束，总投注=%.0f, 总返还=%.0f, 实际RTP=%.4f\n",
		totalBet, totalReturn, totalReturn/totalBet,
	)
	fmt.Printf("耗时: %v\n", time.Since(start))
}
