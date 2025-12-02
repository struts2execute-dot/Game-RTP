# RTP 设计文档

## 一  模式概率

每一次 Spin，我们会得到一次响应结果，大量历史数据排列如下：

| 索引 | 最小投注金额 | 奖励模式 | 中奖倍数 | 牌面数据 |
| ---- | ------------ | -------- | -------- | -------- |
| 0    | 0.1          | normal   | 0        | [reels]  |
| 1    | 0.1          | normal   | 0.5      | [reels]  |
| 2    | 0.1          | normal   | 2        | [reels]  |
| 3    | 0.1          | free     | 8        | [reels]  |
| 4    | 0.1          | cascade  | 1.2      | [reels]  |
| 5    | 0.1          | bonus    | 5        | [reels]  |
| ...  | ......       | ......   | ......   | ......   |

奖励模式：

- normal：普通；
- free：免费旋转；
- cascade：连消；
- bonus：特殊奖励；

将某一个游戏的历史投注结果收集起来，就可以得到上面的数据表格，从而我们可以得到各个奖励模式出现的概率。

| 奖励模式       | 概率 |
| -------------- | ---- |
| normal+cascade | 2/3  |
| free           | 1/6  |
| bonus          | 1/6  |

逆向结果中，各个奖励模式出现的概率应该和原厂保持大致相同，否则会影响用户体验或者导致用户怀疑游戏的真实性，从而影响用户留存。实际操作过程中，可以根据经验来指定各个模式的概率，整体接近即可。

---

## 二 模式 RTP

**设一：整体 RTP 如下：**

$$R_{\text{total}} = 0.97$$

**设二：有如下模式概率配置：**

| 奖励模式       | 概率  | 含义                                     | 设计方式               |
| -------------- | ----- | ---------------------------------------- | ---------------------- |
| normal+cascade | 0.975 | 每 1000 次 spin，有 975 次进入普通模式   | 统计结果 或者 手动设置 |
| free           | 0.012 | 每 1000 次 spin，有 12 次进入 free 大奖  | 统计结果 或者 手动设置 |
| bonus          | 0.013 | 每 1000 次 spin，有 13 次进入 bonus 大奖 | 统计结果 或者 手动设置 |

**设三：有如下模式 RTP 配置：**

| 奖励模式       | RTP  | 含义                                           | 设计方式               |
| -------------- | ---- | ---------------------------------------------- | ---------------------- |
| normal+cascade | 0.6  | 每投注 1 块钱，期望返还 0.6 块钱              | 统计结果 或者 手动设置 |
| free           | 20.0 | 每投注 1 块钱，期望返还 20 块钱               | 统计结果 或者 手动设置 |
| bonus          | ?    | 每投注 1 块钱，期望返还 ? 块钱                | 求解，需满足整体 RTP 0.97 的要求 |

总体平均返还（总体 RTP）是：

- 进入 normal 的概率 × normal 的平均返还；
- 加上进入 free 的概率 × free 的平均返还；
- 加上进入 bonus 的概率 × bonus 的平均返还。

所以整体 RTP 的计算方式就是对**全期望公式**的直接应用：

整体 RTP 公式：

$$
R_\mathrm{total}
= p_\mathrm{normal} \cdot R_\mathrm{normal}
+ p_\mathrm{free} \cdot R_\mathrm{free}
+ p_\mathrm{bonus} \cdot R_\mathrm{bonus}
  $$

反推 bonus 模式的 RTP：

$$
R_\mathrm{bonus}
= \frac{
R_\mathrm{total}
- \bigl(
  p_\mathrm{normal} \cdot R_\mathrm{normal}
    + p_\mathrm{free} \cdot R_\mathrm{free}
      \bigr)
      }{
      p_\mathrm{bonus}
      }
      $$


代入数值得到：

| 奖励模式       | RTP   | 含义                                                                 |
| -------------- | ----- | -------------------------------------------------------------------- |
| normal+cascade | 0.6   | 每投注 1 块钱，期望返还 0.6 块钱（整体呈现亏损状态）                |
| free           | 20.0  | 每投注 1 块钱，期望返还 20 块钱（实际大奖的奖金，来自 normal 亏损的金额） |
| bonus          | 11.15 | 每投注 1 块钱，期望返还 11.15 块钱（实际大奖的奖金，来自 normal 亏损的金额） |

---

## 三 倍数权重

### 3-1 倍数概率

在前面我们已经求出了每个模式的 RTP，那么接下来我们需要进一步理解 RTP 的具体定义，以及 RTP 和中奖倍数的关系。对于第一步“模式概率”中的数据，我们进行维度转换可以得到这样一个结果：

| 模式           | 倍数集合          |
| -------------- | ----------------- |
| normal+cascade | [0, 0.5, 2, 1.2]  |
| free           | [8]               |
| bonus          | [5]               |

很显然，实际计算过程中，这个倍数是不够的，为此，我们模拟一份类似的计算数据，实际过程，通过采集过程筛选出结果即可。以下是模拟数据：

| 模式           | 倍数集合                                                                                                           | rtp   |
| -------------- | ------------------------------------------------------------------------------------------------------------------ | ----- |
| normal+cascade | [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 10, 20]                                                                           | 0.6   |
| free           | [5, 6, 7, 8, 9, 10, 11, 12, 14, 20, 23, 25, 27, 30, 35, 37, 40, 45, 50, 55, 60, 100, 200]                          | 20.0  |
| bonus          | [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 50, 200]                                                          | 11.15 |

单个模式的 RTP 现在是确定的，单个模式的倍数集合我们也已经准备好了。那么接下来我们要了解单个模式下，单个倍数和模式 RTP 之间的数学关系：

$$
R_{m} = \sum_{i=1}^{n} x_i \cdot q_i
$$

- $$x_i$$：第 i 个倍数；
- $$q_i$$：该倍数出现的概率；
- $$R_m$$：该模式的平均返还（模式 RTP）。

根据给出的公式，那么我们接下来要做的是：**求解每一个倍数的概率**。

---

### 3-2 倍数概率求解

#### 3-2-1 符号定义

- $$x_i$$：倍数（第 i 个取值）；
- $$w_i$$：先验权重（与倍数 $$x_i$$ 对应）；
- $$p_i$$：概率（倍数 $$x_i$$ 被采样到的最终概率）；
- $$R_m$$：模式目标 RTP（该模式的平均返还）；
- $$\lambda$$：指数倾斜参数（用于调节分布满足均值约束）；
- $$\lambda^\star$$：最优解的参数；
- $$p_i^\star$$：由 $$\lambda^\star$$ 确定的最终概率分布。

#### 3-2-2 已知与目标

已知：倍数集合、先验权重、目标 RTP

$$
\{x_1, x_2, \dots, x_n\}, \qquad w_i>0, \qquad R_m \text{ (已知)}
$$

目标：求一组概率满足归一与均值约束

$$
\sum_{i=1}^{n} p_i = 1, \qquad
\sum_{i=1}^{n} p_i\,x_i = R_m, \qquad
p_i \ge 0
$$

#### 3-2-3 分布形式（相对先验的最大熵 / 最小 KL）

未归一化权重（指数倾斜）：

$$
p_i \propto w_i \, e^{-\lambda x_i}
$$

归一化后的概率：

$$
p_i(\lambda) =
\frac{w_i \, e^{-\lambda x_i}}
{\sum_{j=1}^{n} w_j \, e^{-\lambda x_j}}
$$

#### 3-2-4 均值约束确定 $$\lambda$$

均值方程：

$$
\sum_{i=1}^{n} p_i(\lambda)\,x_i = R_m
$$

单调性（用二分法求唯一解）：

$$
\frac{d}{d\lambda}\!\left(\sum_{i=1}^{n} p_i(\lambda)\,x_i\right)
= -\,\mathrm{Var}_\lambda(X) \;\le\; 0
$$

#### 3-2-5 最终解

$$
\lambda^\star:\ \sum_{i=1}^{n}
\frac{w_i e^{-\lambda^\star x_i}}
{\sum_{j=1}^{n} w_j e^{-\lambda^\star x_j}}\,x_i
= R_m
$$

$$
p_i^\star
= \frac{w_i \, e^{-\lambda^\star x_i}}
{\sum_{j=1}^{n} w_j \, e^{-\lambda^\star x_j}}
$$

---

### 3-3 波动性调节

波动性越大，RTP 收敛需要的次数则越多，玩起来更刺激；反之则曲线更平滑。

#### 3-3-1 无波动性时（volatility = 1）

$$
p_i(\lambda) =
\frac{e^{-\lambda x_i}}{\sum_j e^{-\lambda x_j}}
$$

#### 3-3-2 加入波动性因子

$$
p_i(\lambda, v) =
\frac{e^{-\lambda x_i}\,(x_i+1)^v}
{\sum_j e^{-\lambda x_j}\,(x_j+1)^v}
$$

在保证目标 RTP 不变的情况下，改变倍数分布的形态：

- 数值越大，分布越厚尾，越容易出大奖；
- 数值越小（甚至负数），分布越集中（收敛靠近模式 RTP），大奖概率下降。

---

## 四 结果验证

### 4-1 计算逻辑

```go
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
```

### 4-2 结果
```text
模式=normal, RTP=0.60
  倍数=0.00, 概率=0.266528
  倍数=0.10, 概率=0.249121
  倍数=0.50, 概率=0.186133
  倍数=1.00, 概率=0.125476
  倍数=1.50, 概率=0.082964
  倍数=2.00, 概率=0.054187
  倍数=3.00, 概率=0.022589
  倍数=4.00, 概率=0.009236
  倍数=5.00, 概率=0.003730
  倍数=10.00, 概率=0.000037
  倍数=20.00, 概率=0.000000
模式=free, RTP=20.00
  倍数=5.00, 概率=0.062726
  倍数=6.00, 概率=0.063388
  倍数=7.00, 概率=0.063662
  倍数=8.00, 概率=0.063635
  倍数=9.00, 概率=0.063372
  倍数=10.00, 概率=0.062921
  倍数=11.00, 概率=0.062316
  倍数=12.00, 概率=0.061589
  倍数=14.00, 概率=0.059855
  倍数=20.00, 概率=0.053429
  倍数=23.00, 概率=0.049957
  倍数=25.00, 概率=0.047641
  倍数=27.00, 概率=0.045350
  倍数=30.00, 概率=0.042001
  倍数=35.00, 概率=0.036738
  倍数=37.00, 概率=0.034762
  倍数=40.00, 概率=0.031947
  倍数=45.00, 概率=0.027656
  倍数=50.00, 概率=0.023856
  倍数=55.00, 概率=0.020519
  倍数=60.00, 概率=0.017607
  倍数=100.00, 概率=0.004902
  倍数=200.00, 概率=0.000169
模式=bonus, RTP=11.15
  倍数=1.00, 概率=0.053167
  倍数=3.00, 概率=0.062405
  倍数=4.00, 概率=0.065152
  倍数=5.00, 概率=0.067192
  倍数=6.00, 概率=0.068713
  倍数=7.00, 概率=0.069835
  倍数=8.00, 概率=0.070640
  倍数=9.00, 概率=0.071189
  倍数=10.00, 概率=0.071527
  倍数=11.00, 概率=0.071687
  倍数=12.00, 概率=0.071697
  倍数=13.00, 概率=0.071580
  倍数=14.00, 概率=0.071354
  倍数=20.00, 概率=0.068402
  倍数=50.00, 概率=0.043624
  倍数=200.00, 概率=0.001835

理论总RTP=0.9700, 标准差σ=3.3757 (volatility=0.30)

模拟结束，总投注=200000, 总返还=194220, 实际RTP=0.9711
耗时: 47.539ms
```
效果：
- 以上是执行20万次spin以后，整体的rtp靠近0.97，那么利润就是（1-0.97）* 总投注。
- 执行次数越多，rtp越趋近于0.97。
- 通过调节volatility，可以控制波动性，volatility控制在[0,2]之间，越小则波动性越大，收敛需要的次数则越多，对应输出中的标准差σ也会越大。
  总结：
- 精准控制rtp，通过调节用户rtp即可控制用户返奖。
- 精准控制模式概率分布和倍数概率分布，游戏体验接近原厂。

---

## 五 spin牌面展示
通过前面的rtp引擎，每次spin，我们通过调用pickMultiplier函数，即可返回一个倍数。
```text
mult := pickMultiplier(multipliers[chosen], modeCDFs[chosen])
```
因此，为了适应spin逻辑，还应该建立模式/倍数和牌面之间的关系。如下：

| 模式   | 倍数 | 牌面                                  |
|--------|------|---------------------------------------|
| normal | 0    | [reels-1,reels-2,reels-3,reels-4]     |
| normal | 1    | [reels-1,reels-2,reels-3,reels-4]     |
| normal | 1.5  | [reels-1,reels-2,reels-3,reels-4]     |
| free   | 10   | [reels-1,reels-2,reels-3,reels-4]     |
| free   | 15   | [reels-1,reels-2,reels-3,reels-4]     |
| bonus  | .... | .....                                 |

每一次spin，我们可以通过随机得到的 模式 + 倍数 得到一个牌面集合（数据存储为hash），然后以随机或者去重且随机的方式任选一个牌面作为返回数据。
最后，根据用户bet的数值，修改实际中奖奖金，返回给前端即可。