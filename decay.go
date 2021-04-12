package main

import (
	"fmt"
	"math"
	"sort"
)

// ================== current decay ==================

func GetExponentialDecayScore(bookedDays int) float64 {
	return 1 * math.Exp(-1*0.001*float64(bookedDays))
}

func testLessThan(val float64, upper float64) string {
	if val >= upper {
		return fmt.Sprintf("ERROR: value %f not less than %f", val, upper)
	}
	return fmt.Sprintf("SUCCESS: value %f less than %f", val, upper)
}

func testGreaterThan(val float64, lower float64) string {
	if val <= lower {
		return fmt.Sprintf("ERROR: value %f not greater than %f", val, lower)
	}
	return fmt.Sprintf("SUCCESS: value %f greater than %f", val, lower)
}

func testIsBetween(val float64, lower float64, upper float64) string {
	if val <= lower || val >= upper {
		return fmt.Sprintf("ERROR: value %f not between %f and %f", val, lower, upper)
	}
	return fmt.Sprintf("SUCCESS: value %f between %f and %f", val, lower, upper)
}

type Decayer interface {
	decay(t float64) float64
}

// ================== exponential decay ==================

type ExponentialDecay struct {
	minValue float64
	factor   float64
}

func InitExponentialDecay(anchorTime uint16, anchorValue float64) ExponentialDecay {
	decay := ExponentialDecay{}
	decay.factor = math.Log(anchorValue) / float64(anchorTime)
	return decay
}

func InitExponentialDecayWithMinConstraint(anchorTime uint16, anchorValue float64, minConstraint float64) ExponentialDecay {
	decay := InitExponentialDecay(anchorTime, anchorValue)
	decay.minValue = minConstraint
	return decay
}

func (d ExponentialDecay) decay(t float64) float64 {
	return math.Min(1, math.Max(math.Exp(d.factor*t), d.minValue))
}

func testExponentialDecay() {
	exponentialDecay := InitExponentialDecayWithMinConstraint(2, 0.5, 0.2)
	fmt.Println("exponentialDecay.decay(1)", testIsBetween(exponentialDecay.decay(1), 0.70, 0.71))
}

// ================== logistic decay ==================

type Anchor struct {
	t float64
	x float64
	y float64
}

type LogisticDecay struct {
	anchors []Anchor
	maxTime float64
}

func InitLogisticDecay(anchors ...Anchor) LogisticDecay {
	decay := LogisticDecay{}
	decay.maxTime = 9999

	decay.anchors = make([]Anchor, len(anchors)+2)
	decay.anchors[0] = Anchor{
		t: 0,
		x: -11.5, // log(1/.99999 - 1)
		y: 1}
	for i := 0; i < len(anchors); i++ {
		anchor := anchors[i]
		if anchor.x == 0 {
			anchor.x = math.Log(1/anchor.y - 1)
		}
		decay.anchors[i+1] = anchor
	}
	decay.anchors[len(decay.anchors)-1] = Anchor{
		t: decay.maxTime,
		x: 11.5, // log(1/.00001 - 1)
		y: 0}
	return decay
}

func InitLogisticDecayWithMaxTime(maxTime float64, anchors ...Anchor) LogisticDecay {
	decay := InitLogisticDecay(anchors...)
	decay.maxTime = maxTime
	return decay
}

func (d LogisticDecay) decay(t float64) float64 {
	T := math.Min(t, d.maxTime)
	i := 1
	for i < len(d.anchors) {
		if T <= d.anchors[i].t {
			break
		}
		i += 1
	}
	anchor_l := d.anchors[i-1]
	anchor_h := d.anchors[i]
	x := anchor_l.x + (T-anchor_l.t)/(anchor_h.t-anchor_l.t)*(anchor_h.x-anchor_l.x)
	return 1 / (1 + math.Exp(x))
}

func testLogisticDecay() {
	logisticDecay := InitLogisticDecay(Anchor{t: 365, y: 0.9}, Anchor{t: 730, y: 0.7}, Anchor{t: 1825, y: 0.25})
	fmt.Println("logisticDecay.decay(0)", testGreaterThan(logisticDecay.decay(0), 0.9999))
	fmt.Println("logisticDecay.decay(180)", testIsBetween(logisticDecay.decay(180), 0.9, 0.9999))
	fmt.Println("logisticDecay.decay(365)", testIsBetween(logisticDecay.decay(365), 0.89, 0.91))
	fmt.Println("logisticDecay.decay(500)", testIsBetween(logisticDecay.decay(500), 0.7, 0.9))
	fmt.Println("logisticDecay.decay(730)", testIsBetween(logisticDecay.decay(730), 0.69, 0.71))
	fmt.Println("logisticDecay.decay(1000)", testIsBetween(logisticDecay.decay(1000), 0.25, 0.7))
	fmt.Println("logisticDecay.decay(1825)", testIsBetween(logisticDecay.decay(1825), 0.24, 0.26))
	fmt.Println("logisticDecay.decay(2000)", testIsBetween(logisticDecay.decay(2000), 0, 0.25))
}

// ================== piecewise linear decay ==================

type LinearFunction struct {
	m float64
	c float64
}

func (l LinearFunction) compute(x float64) float64 {
	return l.m*x + l.c
}

type PiecewiseRange struct {
	lower    float64
	upper    float64
	function LinearFunction
}

type PiecewiseLinearDecay struct {
	ranges []PiecewiseRange
}

func InitPiecewiseLinearDecay(anchors ...Anchor) PiecewiseLinearDecay {
	if anchors[0].x != 0 {
		anchors = append([]Anchor{Anchor{x: 0, y: 1}}, anchors...)
	}
	sort.Slice(anchors, func(i, j int) bool {
		return anchors[i].x < anchors[j].x
	})
	ranges := make([]PiecewiseRange, len(anchors))
	var anchor_l, anchor_h Anchor
	i := 0
	for ; i < (len(anchors) - 1); i++ {
		anchor_l = anchors[i]
		anchor_h = anchors[i+1]
		ranges[i] = getPiecewiseRange(anchor_l, anchor_h)
	}
	ranges[i] = getPiecewiseRange(anchor_h, Anchor{x: math.Inf(0), y: anchor_h.y})

	return PiecewiseLinearDecay{ranges}
}

func getPiecewiseRange(anchor_l Anchor, anchor_h Anchor) PiecewiseRange {
	r := PiecewiseRange{}
	r.lower = anchor_l.x
	r.upper = anchor_h.x
	r.function = getLinearFunction(anchor_l, anchor_h)
	return r

}

func getLinearFunction(anchor_l Anchor, anchor_h Anchor) LinearFunction {
	lf := LinearFunction{}
	lf.m = (anchor_h.y - anchor_l.y) / (anchor_h.x - anchor_l.x)
	lf.c = anchor_l.y - (lf.m * anchor_l.x)
	return lf
}

func (d PiecewiseLinearDecay) decay(t float64) float64 {
	for i := 0; i < len(d.ranges); i++ {
		if t >= d.ranges[i].lower && t <= d.ranges[i].upper {
			return d.ranges[i].function.compute(t)
		}
	}
	return 0
}

func testPiecewiseLinearDecay() {
	piecewiseLinearDecay := InitPiecewiseLinearDecay(Anchor{x: 90, y: 0.99}, Anchor{x: 180, y: 0.5}, Anchor{x: 270, y: 0.5}, Anchor{x: 365, y: 0.75}, Anchor{x: 366, y: 0.1})
	fmt.Println("piecewiseLinearDecay.decay(0)", testGreaterThan(piecewiseLinearDecay.decay(0), 0.9999))
	fmt.Println("piecewiseLinearDecay.decay(45)", testIsBetween(piecewiseLinearDecay.decay(45), 0.99, 1))
	fmt.Println("piecewiseLinearDecay.decay(90)", testIsBetween(piecewiseLinearDecay.decay(90), 0.98, 0.999))
	fmt.Println("piecewiseLinearDecay.decay(100)", testIsBetween(piecewiseLinearDecay.decay(100), 0.5, 0.99))
	fmt.Println("piecewiseLinearDecay.decay(180)", testIsBetween(piecewiseLinearDecay.decay(180), 0.49, 0.51))
	fmt.Println("piecewiseLinearDecay.decay(300)", testIsBetween(piecewiseLinearDecay.decay(300), 0.5, 0.75))
	fmt.Println("piecewiseLinearDecay.decay(365)", testIsBetween(piecewiseLinearDecay.decay(365), 0.749, 0.751))
	fmt.Println("piecewiseLinearDecay.decay(400)", testIsBetween(piecewiseLinearDecay.decay(400), 0.09, 0.11))
}

// ================== main ==================

var piecewiseLinearDecay = InitPiecewiseLinearDecay(Anchor{x: 1, y: 0.95}, Anchor{x: 2, y: 0.9}, Anchor{x: 7, y: 0.3}, Anchor{x: 8, y: 0.01})

func decay(t float64, algo string) float64 {
	switch algo {
	case "current":
		return GetExponentialDecayScore(int(t))
	case "pld":
		return piecewiseLinearDecay.decay(t)
	default:
		return GetExponentialDecayScore(int(t))
	}
}

func main() {
	testExponentialDecay()
	testLogisticDecay()
	testPiecewiseLinearDecay()

	fmt.Println("current decay", decay(2, "current"))
	fmt.Println("new decay", decay(2, "pld"))
}