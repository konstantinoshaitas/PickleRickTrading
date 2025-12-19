package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"
)

// ============================================================================
// 1. DATA STRUCTURES
// ============================================================================

type MarketData struct {
	Time  []int64
	Open  []float64
	Close []float64
}

func (md *MarketData) Slice(start, end int) *MarketData {
	return &MarketData{
		Time:  md.Time[start:end],
		Open:  md.Open[start:end],
		Close: md.Close[start:end],
	}
}

// StrategyType Enum
type StrategyType int

const (
	StratMACD StrategyType = iota // Uses EMA (Classic MACD)
	StratTEMA                     // Uses TEMA (TEMA Oscillator)
	Strat3EMA                     // Uses 3EMA (TRIX-like)
	StratKAMA                     // Uses KAMA (Adaptive)
)

func (s StrategyType) String() string {
	return [...]string{"MACD (EMA)", "TEMA", "3EMA", "KAMA"}[s]
}

// Params holds the inputs
type StrategyParams struct {
	Type   StrategyType `json:"Type"`
	Fast   int          `json:"F"`
	Slow   int          `json:"S"`
	Signal int          `json:"Sig"`
}

// PerformanceMetrics holds the advanced stats
type PerformanceMetrics struct {
	TotalTrades int     `json:"Total_Trades"`
	WinRate     float64 `json:"Win_Rate"`
	CAGR        float64 `json:"CAGR"`
	Sharpe      float64 `json:"Sharpe"`
	Calmar      float64 `json:"Calmar"`
	Ulcer       float64 `json:"Ulcer"`
	Martin      float64 `json:"Martin"`
	MaxDD       float64 `json:"Max_DD"`
}

// StabilityMetrics holds neighbor analysis
type StabilityMetrics struct {
	LocalMean      float64 `json:"Local_Mean"`
	LocalStDev     float64 `json:"Local_StDev"`
	StabilityRatio float64 `json:"Stability_Ratio"`
}

// FullResult is the master object exported to JSON
type FullResult struct {
	Params    StrategyParams     `json:"Params"`
	Metrics   PerformanceMetrics `json:"Metrics"`
	Stability StabilityMetrics   `json:"Stability"`
}

// BacktestBuffers - Generic buffer for calculations
type BacktestBuffers struct {
	MainLine   []float64 // The Oscillator (Fast - Slow)
	SignalLine []float64 // The Signal (EMA of MainLine)
	Hist       []float64 // The Histogram (Main - Signal)
	Equity     []float64 // Track curve

	// Temporary scratch buffers used for intermediate MA calcs
	TmpFast []float64
	TmpSlow []float64
	TmpMisc []float64
}

type DailyRecord struct {
	Date     string
	Phase    string
	Price    float64
	Action   string
	Cash     float64
	Holdings float64
	Equity   float64
}

// ============================================================================
// 2. MEMORY POOL
// ============================================================================

var bufferPool *sync.Pool

func InitPool(size int) {
	bufferPool = &sync.Pool{
		New: func() interface{} {
			return &BacktestBuffers{
				MainLine:   make([]float64, size),
				SignalLine: make([]float64, size),
				Hist:       make([]float64, size),
				Equity:     make([]float64, size),
				TmpFast:    make([]float64, size),
				TmpSlow:    make([]float64, size),
				TmpMisc:    make([]float64, size),
			}
		},
	}
}

// ============================================================================
// 3. PURE INDICATOR LOGIC
// ============================================================================

// IndicatorFunc defines the signature for any indicator calculator
type IndicatorFunc func(data []float64, p StrategyParams, buf *BacktestBuffers)

// Router
func CalculateIndicator(data []float64, p StrategyParams, buf *BacktestBuffers) {
	switch p.Type {
	case StratMACD:
		CalcMACD_Standard(data, p.Fast, p.Slow, p.Signal, buf)
	case StratTEMA:
		CalcTEMA_Oscillator(data, p.Fast, p.Slow, p.Signal, buf)
	case Strat3EMA:
		Calc3EMA_Oscillator(data, p.Fast, p.Slow, p.Signal, buf)
	case StratKAMA:
		CalcKAMA_Oscillator(data, p.Fast, p.Slow, p.Signal, buf)
	}
}

// --- CORE MATH PRIMITIVES ---

// 1. Standard EMA
func CalcEMA(data []float64, period int, out []float64) {
	k := 2.0 / float64(period+1)
	out[0] = data[0]
	for i := 1; i < len(data); i++ {
		out[i] = data[i]*k + out[i-1]*(1-k)
	}
}

// 2. TEMA Primitive (Triple Exponential Moving Average)
func CalcTEMA_Single(data []float64, period int, out, t1, t2, t3 []float64) {
	CalcEMA(data, period, t1) // EMA1
	CalcEMA(t1, period, t2)   // EMA2 of EMA1
	CalcEMA(t2, period, t3)   // EMA3 of EMA2

	for i := 0; i < len(data); i++ {
		// TEMA = 3*EMA1 - 3*EMA2 + EMA3
		out[i] = 3*t1[i] - 3*t2[i] + t3[i]
	}
}

// 3. 3EMA Primitive (Triple Consecutive EMA)
// This is NOT TEMA. This is EMA(EMA(EMA(x))). Used in TRIX.
func Calc3EMA_Single(data []float64, period int, out, t1, t2 []float64) {
	CalcEMA(data, period, t1)
	CalcEMA(t1, period, t2)
	CalcEMA(t2, period, out)
}

// 4. KAMA Primitive (Kaufman Adaptive)
func CalcKAMA_Single(data []float64, period int, out []float64) {
	n := len(data)
	if n == 0 {
		return
	}
	erPeriod := period
	fastest := 2.0 / (2.0 + 1.0)
	slowest := 2.0 / (30.0 + 1.0)

	out[0] = data[0]
	for i := 1; i < n; i++ {
		if i < erPeriod {
			out[i] = data[i]
			continue
		}
		change := math.Abs(data[i] - data[i-erPeriod])
		volatility := 0.0
		for j := 0; j < erPeriod; j++ {
			volatility += math.Abs(data[i-j] - data[i-j-1])
		}
		er := 0.0
		if volatility != 0 {
			er = change / volatility
		}
		sc := math.Pow(er*(fastest-slowest)+slowest, 2)
		out[i] = out[i-1] + sc*(data[i]-out[i-1])
	}
}

// --- STRATEGY IMPLEMENTATIONS ---

// A. STANDARD MACD (EMA based)
func CalcMACD_Standard(data []float64, fast, slow, signal int, buf *BacktestBuffers) {
	// 1. Calculate Fast EMA
	CalcEMA(data, fast, buf.TmpFast)
	// 2. Calculate Slow EMA
	CalcEMA(data, slow, buf.TmpSlow)

	// 3. Oscillator = Fast - Slow
	for i := 0; i < len(data); i++ {
		buf.MainLine[i] = buf.TmpFast[i] - buf.TmpSlow[i]
	}

	// 4. Signal Line = EMA of Oscillator
	CalcEMA(buf.MainLine, signal, buf.SignalLine)

	// 5. Histogram
	for i := 0; i < len(data); i++ {
		buf.Hist[i] = buf.MainLine[i] - buf.SignalLine[i]
	}
}

// B. TEMA OSCILLATOR (Pure TEMA)
func CalcTEMA_Oscillator(data []float64, fast, slow, signal int, buf *BacktestBuffers) {
	// We use Equity/Hist buffers as scratch space for the recursion of TEMA calcs
	// This avoids allocating new memory.

	// 1. Calculate Fast TEMA -> Stored in TmpFast
	CalcTEMA_Single(data, fast, buf.TmpFast, buf.Equity, buf.SignalLine, buf.Hist)

	// 2. Calculate Slow TEMA -> Stored in TmpSlow
	CalcTEMA_Single(data, slow, buf.TmpSlow, buf.Equity, buf.SignalLine, buf.Hist)

	// Reset scratch buffers (Equity/Signal/Hist) to clean state if needed,
	// though we overwrite them below anyway.

	// 3. Oscillator = Fast TEMA - Slow TEMA
	for i := 0; i < len(data); i++ {
		buf.MainLine[i] = buf.TmpFast[i] - buf.TmpSlow[i]
	}

	// 4. Signal Line = EMA of Oscillator
	CalcEMA(buf.MainLine, signal, buf.SignalLine)

	// 5. Histogram
	for i := 0; i < len(data); i++ {
		buf.Hist[i] = buf.MainLine[i] - buf.SignalLine[i]
	}

	// Clean Equity for actual use later
	for i := range buf.Equity {
		buf.Equity[i] = 0
	}
}

// C. 3EMA OSCILLATOR (Pure 3EMA)
func Calc3EMA_Oscillator(data []float64, fast, slow, signal int, buf *BacktestBuffers) {
	// 1. Fast 3EMA
	Calc3EMA_Single(data, fast, buf.TmpFast, buf.TmpMisc, buf.Hist)
	// 2. Slow 3EMA
	Calc3EMA_Single(data, slow, buf.TmpSlow, buf.TmpMisc, buf.Hist)

	// 3. Oscillator
	for i := range data {
		buf.MainLine[i] = buf.TmpFast[i] - buf.TmpSlow[i]
	}

	// 4. Signal
	CalcEMA(buf.MainLine, signal, buf.SignalLine)

	// 5. Hist
	for i := range data {
		buf.Hist[i] = buf.MainLine[i] - buf.SignalLine[i]
	}
}

// D. KAMA OSCILLATOR (Pure KAMA)
func CalcKAMA_Oscillator(data []float64, fast, slow, signal int, buf *BacktestBuffers) {
	// 1. Fast KAMA
	CalcKAMA_Single(data, fast, buf.TmpFast)
	// 2. Slow KAMA
	CalcKAMA_Single(data, slow, buf.TmpSlow)

	// 3. Oscillator
	for i := range data {
		buf.MainLine[i] = buf.TmpFast[i] - buf.TmpSlow[i]
	}

	// 4. Signal
	CalcEMA(buf.MainLine, signal, buf.SignalLine)

	// 5. Hist
	for i := range data {
		buf.Hist[i] = buf.MainLine[i] - buf.SignalLine[i]
	}
}

// ============================================================================
// 4. METRIC CALCULATION ENGINE
// ============================================================================

func CalculateMetrics(equityCurve []float64, trades int, wins int) PerformanceMetrics {
	n := len(equityCurve)
	if n == 0 {
		return PerformanceMetrics{}
	}

	startEquity := equityCurve[0]
	endEquity := equityCurve[n-1]

	// 1. CAGR
	years := float64(n) / 365.0
	cagr := 0.0
	if endEquity > 0 && startEquity > 0 {
		cagr = math.Pow(endEquity/startEquity, 1/years) - 1
	}

	// 2. Drawdowns & Ulcer Index
	peak := startEquity
	sumSqDD := 0.0
	maxDD := 0.0
	returns := make([]float64, 0, n)

	for i, val := range equityCurve {
		if val > peak {
			peak = val
		}
		dd := (val - peak) / peak
		if dd < maxDD {
			maxDD = dd
		}
		sumSqDD += dd * dd

		if i > 0 && equityCurve[i-1] != 0 {
			ret := (equityCurve[i] - equityCurve[i-1]) / equityCurve[i-1]
			returns = append(returns, ret)
		}
	}

	ulcer := math.Sqrt(sumSqDD / float64(n))

	// 3. Martin Ratio
	martin := 0.0
	if ulcer > 0 {
		martin = cagr / ulcer
	}

	// 4. Sharpe Ratio
	meanRet := 0.0
	for _, r := range returns {
		meanRet += r
	}
	if len(returns) > 0 {
		meanRet /= float64(len(returns))
	}

	variance := 0.0
	for _, r := range returns {
		variance += math.Pow(r-meanRet, 2)
	}

	stdDev := 0.0
	sharpe := 0.0
	if len(returns) > 0 {
		stdDev = math.Sqrt(variance / float64(len(returns)))
		if stdDev > 0 {
			sharpe = (meanRet / stdDev) * math.Sqrt(365)
		}
	}

	// 5. Calmar Ratio
	calmar := 0.0
	if maxDD != 0 {
		calmar = cagr / math.Abs(maxDD)
	}

	winRate := 0.0
	if trades > 0 {
		winRate = float64(wins) / float64(trades)
	}

	return PerformanceMetrics{
		TotalTrades: trades,
		WinRate:     winRate,
		CAGR:        cagr,
		MaxDD:       maxDD,
		Ulcer:       ulcer,
		Martin:      martin,
		Sharpe:      sharpe,
		Calmar:      calmar,
	}
}

// ============================================================================
// 5. BACKTEST ENGINE
// ============================================================================

func RunBacktest(data *MarketData, p StrategyParams) FullResult {
	if p.Fast >= p.Slow {
		return FullResult{Params: p}
	}

	bufs := bufferPool.Get().(*BacktestBuffers)
	subBuf := &BacktestBuffers{
		MainLine:   bufs.MainLine[:len(data.Close)],
		SignalLine: bufs.SignalLine[:len(data.Close)],
		Hist:       bufs.Hist[:len(data.Close)],
		Equity:     bufs.Equity[:len(data.Close)],
		TmpFast:    bufs.TmpFast[:len(data.Close)],
		TmpSlow:    bufs.TmpSlow[:len(data.Close)],
		TmpMisc:    bufs.TmpMisc[:len(data.Close)],
	}
	defer bufferPool.Put(bufs)

	// --- MODULAR INDICATOR CALL ---
	CalculateIndicator(data.Close, p, subBuf)
	// ------------------------------

	cash := 10000.0
	position := 0.0
	trades := 0
	wins := 0

	hist := subBuf.Hist
	equityCurve := subBuf.Equity
	equityCurve[0] = 10000.0

	entryPrice := 0.0

	for i := 1; i < len(data.Close)-1; i++ {
		prevHist := hist[i-1]
		currHist := hist[i]
		nextOpen := data.Open[i+1]

		// Logic: Crossover of Histogram (which means MainLine crossed SignalLine)
		if position == 0 && prevHist < 0 && currHist > 0 {
			// BUY
			position = cash / nextOpen
			entryPrice = nextOpen
			cash = 0
			trades++
		} else if position > 0 && prevHist > 0 && currHist < 0 {
			// SELL
			exitVal := position * nextOpen
			if exitVal > (position * entryPrice) {
				wins++
			}
			cash = exitVal
			position = 0
		}

		if position > 0 {
			equityCurve[i] = position * data.Close[i]
		} else {
			equityCurve[i] = cash
		}
	}

	lastIdx := len(data.Close) - 1
	if position > 0 {
		equityCurve[lastIdx] = position * data.Close[lastIdx]
	} else {
		equityCurve[lastIdx] = cash
	}

	metrics := CalculateMetrics(equityCurve, trades, wins)

	return FullResult{
		Params:  p,
		Metrics: metrics,
	}
}

func RunDetailedBacktest(data *MarketData, p StrategyParams, splitIndex int) []DailyRecord {
	bufs := bufferPool.Get().(*BacktestBuffers)
	defer bufferPool.Put(bufs)

	CalculateIndicator(data.Close, p, bufs)

	cash := 10000.0
	position := 0.0
	hist := bufs.Hist

	records := make([]DailyRecord, 0, len(data.Close))
	records = append(records, DailyRecord{
		Date:   msToTime(data.Time[0]).Format("2006-01-02"),
		Phase:  "IS",
		Price:  data.Close[0],
		Cash:   10000.0,
		Equity: 10000.0,
	})

	for i := 1; i < len(data.Close)-1; i++ {
		phase := "IS"
		if i >= splitIndex {
			phase = "OOS"
		}

		prevHist := hist[i-1]
		currHist := hist[i]
		nextIdx := i + 1
		nextOpen := data.Open[nextIdx]
		nextDate := msToTime(data.Time[nextIdx]).Format("2006-01-02")

		action := ""
		if position == 0 && prevHist < 0 && currHist > 0 {
			position = cash / nextOpen
			cash = 0
			action = "BUY"
		} else if position > 0 && prevHist > 0 && currHist < 0 {
			cash = position * nextOpen
			position = 0
			action = "SELL"
		}

		currentEquity := cash
		currentHoldings := 0.0
		if position > 0 {
			currentHoldings = position * data.Close[nextIdx]
			currentEquity = currentHoldings
		}

		records = append(records, DailyRecord{
			Date:     nextDate,
			Phase:    phase,
			Price:    data.Close[nextIdx],
			Action:   action,
			Holdings: currentHoldings,
			Cash:     cash,
			Equity:   currentEquity,
		})
	}
	return records
}

// ============================================================================
// 6. GRID SEARCH & STABILITY ENGINE
// ============================================================================

func RunGridSearch(trainData *MarketData, stratType StrategyType) []FullResult {
	log.Println("------------------------------------------------")
	log.Printf("STARTING GRID SEARCH: %s\n", stratType.String())
	log.Printf("Threads: %d | Train Candles: %d\n", runtime.NumCPU(), len(trainData.Close))
	log.Println("------------------------------------------------")

	// 1. Grid Ranges
	fastRange := make([]int, 0)
	for i := 1; i <= 300; i++ {
		fastRange = append(fastRange, i)
	}

	slowRange := make([]int, 0)
	for i := 1; i <= 300; i++ {
		slowRange = append(slowRange, i)
	}

	sigRange := make([]int, 0)
	for i := 1; i <= 300; i++ {
		sigRange = append(sigRange, i)
	}

	// Maps for 3D Cube lookup
	fastMap := make(map[int]int)
	for i, v := range fastRange {
		fastMap[v] = i
	}
	slowMap := make(map[int]int)
	for i, v := range slowRange {
		slowMap[v] = i
	}
	sigMap := make(map[int]int)
	for i, v := range sigRange {
		sigMap[v] = i
	}

	// 2. Initialize 3D Cube
	cube := make([][][]float64, len(fastRange))
	for i := range cube {
		cube[i] = make([][]float64, len(slowRange))
		for j := range cube[i] {
			cube[i][j] = make([]float64, len(sigRange))
			for k := range cube[i][j] {
				cube[i][j][k] = -999.0
			}
		}
	}

	// 3. Worker Pool
	jobs := make(chan StrategyParams, 5000)
	resultsChan := make(chan FullResult, 5000)
	var wg sync.WaitGroup

	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range jobs {
				resultsChan <- RunBacktest(trainData, p)
			}
		}()
	}

	go func() {
		for _, f := range fastRange {
			for _, s := range slowRange {
				if f >= s {
					continue
				}
				for _, sig := range sigRange {
					jobs <- StrategyParams{stratType, f, s, sig}
				}
			}
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// 4. Collect & Filter
	var validResults []FullResult
	const MinTrades = 8

	for res := range resultsChan {
		if res.Metrics.TotalTrades > 0 {
			fIdx := fastMap[res.Params.Fast]
			sIdx := slowMap[res.Params.Slow]
			sigIdx := sigMap[res.Params.Signal]
			cube[fIdx][sIdx][sigIdx] = res.Metrics.Martin
		}

		if res.Metrics.TotalTrades >= MinTrades {
			validResults = append(validResults, res)
		}
	}

	log.Printf("Strategies evaluated: %d. Valid candidates: %d.\n",
		len(fastRange)*len(slowRange)*len(sigRange)/2, len(validResults))

	// 5. Calculate Stability
	log.Println("Computing Stability Metrics...")

	for i := range validResults {
		res := &validResults[i]
		fIdx := fastMap[res.Params.Fast]
		sIdx := slowMap[res.Params.Slow]
		sigIdx := sigMap[res.Params.Signal]

		vals := make([]float64, 0, 27)

		for dx := -1; dx <= 1; dx++ {
			for dy := -1; dy <= 1; dy++ {
				for dz := -1; dz <= 1; dz++ {
					nx, ny, nz := fIdx+dx, sIdx+dy, sigIdx+dz

					if nx >= 0 && nx < len(fastRange) &&
						ny >= 0 && ny < len(slowRange) &&
						nz >= 0 && nz < len(sigRange) {

						val := cube[nx][ny][nz]
						if val > -100 {
							vals = append(vals, val)
						}
					}
				}
			}
		}

		if len(vals) < 3 {
			res.Stability.StabilityRatio = 0
			continue
		}

		sum := 0.0
		for _, v := range vals {
			sum += v
		}
		mean := sum / float64(len(vals))

		variance := 0.0
		for _, v := range vals {
			variance += math.Pow(v-mean, 2)
		}
		stdDev := math.Sqrt(variance / float64(len(vals)))

		res.Stability.LocalMean = mean
		res.Stability.LocalStDev = stdDev

		if stdDev < 0.000001 {
			res.Stability.StabilityRatio = 20.0
		} else {
			res.Stability.StabilityRatio = mean / stdDev
		}
	}

	sort.Slice(validResults, func(i, j int) bool {
		return validResults[i].Stability.StabilityRatio > validResults[j].Stability.StabilityRatio
	})

	// Export cube for 3D visualization
	SaveCubeJSON(cube, fastRange, slowRange, sigRange, "cube_data.json")

	return validResults
}

// ============================================================================
// 7. UTILS & DATA
// ============================================================================

func DownloadCryptoData(symbol string, startDate string) (*MarketData, error) {
	log.Printf("Fetching full history for %s...", symbol)
	t, _ := time.Parse("2006-01-02", startDate)
	startTime := t.UnixMilli()

	data := &MarketData{
		Time:  make([]int64, 0),
		Open:  make([]float64, 0),
		Close: make([]float64, 0),
	}

	client := &http.Client{Timeout: 10 * time.Second}

	for {
		url := fmt.Sprintf("https://api.binance.com/api/v3/klines?symbol=%s&interval=1d&limit=1000&startTime=%d", symbol, startTime)
		resp, err := client.Get(url)
		if err != nil {
			return nil, err
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		var rawData [][]interface{}
		if err := json.Unmarshal(body, &rawData); err != nil {
			return nil, err
		}

		if len(rawData) == 0 {
			break
		}

		for _, kline := range rawData {
			ts := int64(kline[0].(float64))
			o, _ := strconv.ParseFloat(kline[1].(string), 64)
			c, _ := strconv.ParseFloat(kline[4].(string), 64)

			data.Time = append(data.Time, ts)
			data.Open = append(data.Open, o)
			data.Close = append(data.Close, c)
		}

		lastTs := int64(rawData[len(rawData)-1][0].(float64))
		startTime = lastTs + 1

		fmt.Printf("\rDownloaded %d candles...", len(data.Close))
		if len(rawData) < 1000 {
			break
		}
	}
	fmt.Println()
	if len(data.Close) == 0 {
		return nil, fmt.Errorf("No data found")
	}
	return data, nil
}

func ExportToCSV(records []DailyRecord, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"Date", "Phase", "Price", "Action", "Cash", "Holdings", "Equity"})
	for _, r := range records {
		writer.Write([]string{
			r.Date, r.Phase, fmt.Sprintf("%.2f", r.Price), r.Action,
			fmt.Sprintf("%.2f", r.Cash), fmt.Sprintf("%.2f", r.Holdings), fmt.Sprintf("%.2f", r.Equity),
		})
	}
	return nil
}

func SaveResultsJSON(results []FullResult, filename string) {
	file, _ := os.Create(filename)
	defer file.Close()
	encoder := json.NewEncoder(file)
	encoder.Encode(results)
}

type CubePoint struct {
	Fast   int     `json:"Fast"`
	Slow   int     `json:"Slow"`
	Signal int     `json:"Signal"`
	Martin float64 `json:"Martin"`
}

func SaveCubeJSON(cube [][][]float64, fastRange, slowRange, sigRange []int, filename string) {
	var points []CubePoint
	for i, f := range fastRange {
		for j, s := range slowRange {
			for k, sig := range sigRange {
				val := cube[i][j][k]
				if val > -100 {
					points = append(points, CubePoint{
						Fast: f, Slow: s, Signal: sig, Martin: val,
					})
				}
			}
		}
	}

	file, _ := os.Create(filename)
	defer file.Close()
	json.NewEncoder(file).Encode(points)
	log.Printf("Saved cube data to '%s' (%d points)\n", filename, len(points))
}

func msToTime(ms int64) time.Time {
	return time.Unix(0, ms*int64(time.Millisecond))
}

// ============================================================================
// 8. MAIN EXECUTION
// ============================================================================

func main() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)

	// --- CONFIGURATION ---
	// Options: StratMACD, StratTEMA, Strat3EMA, StratKAMA
	selectedStrategy := StratMACD
	symbol := "BTCUSDT"
	// ---------------------

	// 1. Download Data
	data, err := DownloadCryptoData(symbol, "2018-01-01")
	if err != nil {
		log.Fatal("Data download failed:", err)
	}

	// 2. Split Data
	splitRatio := 0.70
	splitIndex := int(float64(len(data.Close)) * splitRatio)
	InitPool(len(data.Close))
	trainData := data.Slice(0, splitIndex)

	fmt.Printf("Train Data: %d candles | Test Data: %d candles\n", len(trainData.Close), len(data.Close)-splitIndex)

	// 3. Run Optimization
	startOpt := time.Now()

	// Pass the selected strategy type here
	results := RunGridSearch(trainData, selectedStrategy)

	if len(results) == 0 {
		log.Fatal("No valid strategies found. Check data or lower MinTrades constraint.")
	}

	// 4. Select Best Stable Strategy
	best := results[0]

	fmt.Println("\n================================================")
	fmt.Println("OPTIMIZATION COMPLETE")
	fmt.Printf("Strategy: %s\n", selectedStrategy.String())
	fmt.Printf("Time: %s\n", time.Since(startOpt))
	fmt.Printf("Valid Candidates: %d\n", len(results))
	fmt.Println("------------------------------------------------")
	fmt.Println("TOP STRATEGY (Highest Stability):")
	fmt.Printf("Params: Fast=%d Slow=%d Sig=%d\n", best.Params.Fast, best.Params.Slow, best.Params.Signal)
	fmt.Printf("Stability Ratio: %.2f\n", best.Stability.StabilityRatio)
	fmt.Printf("Martin Ratio:    %.2f\n", best.Metrics.Martin)
	fmt.Printf("CAGR:            %.2f%%\n", best.Metrics.CAGR*100)
	fmt.Printf("Max Drawdown:    %.2f%%\n", best.Metrics.MaxDD*100)
	fmt.Printf("Win Rate:        %.2f%%\n", best.Metrics.WinRate*100)
	fmt.Println("================================================")

	// 5. Save Grid Results for Python Plotting
	SaveResultsJSON(results, "grid_results.json")
	fmt.Println("Saved 'grid_results.json'")

	// 6. Run Detailed Backtest for Best Strategy
	fmt.Println("Generating detailed equity curve for Best Strategy...")
	records := RunDetailedBacktest(data, best.Params, splitIndex)

	csvName := "best_strategy_equity.csv"
	ExportToCSV(records, csvName)
	fmt.Printf("Saved '%s'\n", csvName)

	// 7. Check OOS Performance
	startOOS := records[splitIndex].Equity
	endOOS := records[len(records)-1].Equity
	oosRet := (endOOS - startOOS) / startOOS * 100
	fmt.Printf(">>> OOS RETURN: %.2f%% <<<\n", oosRet)
}
