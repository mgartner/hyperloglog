package hyperloglog

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"slices"
)

const (
	pp      = uint8(25)
	mp      = uint32(1) << pp
	version = 2
)

type Sketch struct {
	p          uint8
	m          uint32
	alpha      float64
	tmpSet     set
	sparseList *compressedList
	regs       []uint8
}

// New returns a HyperLogLog Sketch with 2^14 registers (precision 14)
func New() *Sketch { return New14() }

// New14 returns a HyperLogLog Sketch with 2^14 registers (precision 14)
func New14() *Sketch { return newSketchNoError(14, true) }

// New16 returns a HyperLogLog Sketch with 2^16 registers (precision 16)
func New16() *Sketch { return newSketchNoError(16, true) }

// NewNoSparse returns a HyperLogLog Sketch with 2^14 registers (precision 14) that will not use a sparse representation
func NewNoSparse() *Sketch { return newSketchNoError(14, false) }

// New16NoSparse returns a HyperLogLog Sketch with 2^16 registers (precision 16) that will not use a sparse representation
func New16NoSparse() *Sketch { return newSketchNoError(16, false) }

func newSketchNoError(precision uint8, sparse bool) *Sketch {
	sk, _ := NewSketch(precision, sparse)
	return sk
}

func NewSketch(precision uint8, sparse bool) (*Sketch, error) {
	if precision < 4 || precision > 18 {
		return nil, fmt.Errorf("p has to be >= 4 and <= 18")
	}
	m := uint32(1) << precision
	s := &Sketch{
		m:     m,
		p:     precision,
		alpha: alpha(float64(m)),
	}
	if sparse {
		s.tmpSet = makeSet(0)
		s.sparseList = newCompressedList(0)
	} else {
		s.regs = make([]uint8, m)
	}
	return s, nil
}

func (sk *Sketch) InitDenseSketch(precision uint8, registers []uint8) error {
	if precision < 4 || precision > 18 {
		return fmt.Errorf("precision must be >= 4 and <= 18")
	}
	m := uint32(1) << precision
	if len(registers) != int(m) {
		return fmt.Errorf("expected %d registers", m)
	}
	sk.m = m
	sk.p = precision
	sk.alpha = alpha(float64(m))
	sk.regs = registers
	return nil
}

type Sketch10Dense struct {
	Sketch
	regs [1 << 10]uint8
}

func (sk *Sketch10Dense) Init() {
	const precision = 10
	sk.m = uint32(1) << precision
	sk.p = precision
	sk.alpha = alpha(precision)
	sk.Sketch.regs = sk.regs[:]
}

func (sk *Sketch) sparse() bool { return sk.sparseList != nil }

// Clone returns a deep copy of sk.
func (sk *Sketch) Clone() *Sketch {
	clone := *sk
	clone.regs = append([]uint8(nil), sk.regs...)
	clone.tmpSet = sk.tmpSet.Clone()
	clone.sparseList = sk.sparseList.Clone()
	return &clone
}

func (sk *Sketch) maybeToNormal() {
	if uint32(sk.tmpSet.Len())*100 > sk.m {
		sk.mergeSparse()
		if uint32(sk.sparseList.Len()) > sk.m {
			sk.toNormal()
		}
	}
}

func (sk *Sketch) Merge(other *Sketch) error {
	if other == nil {
		return nil
	}
	if sk.p != other.p {
		return errors.New("precisions must be equal")
	}

	if sk.sparse() && other.sparse() {
		sk.mergeSparseSketch(other)
	} else {
		sk.mergeDenseSketch(other)
	}
	return nil
}

func (sk *Sketch) mergeSparseSketch(other *Sketch) {
	sk.tmpSet.Merge(other.tmpSet)
	for iter := other.sparseList.Iter(); iter.HasNext(); {
		sk.tmpSet.add(iter.Next())
	}
	sk.maybeToNormal()
}

func (sk *Sketch) mergeDenseSketch(other *Sketch) {
	if sk.sparse() {
		sk.toNormal()
	}

	if other.sparse() {
		other.tmpSet.ForEach(func(k uint32) {
			i, r := decodeHash(k, other.p, pp)
			sk.insert(i, r)
		})
		for iter := other.sparseList.Iter(); iter.HasNext(); {
			i, r := decodeHash(iter.Next(), other.p, pp)
			sk.insert(i, r)
		}
	} else {
		for i, v := range other.regs {
			if v > sk.regs[i] {
				sk.regs[i] = v
			}
		}
	}
}

func (sk *Sketch) toNormal() {
	if sk.tmpSet.Len() > 0 {
		sk.mergeSparse()
	}

	sk.regs = make([]uint8, sk.m)
	for iter := sk.sparseList.Iter(); iter.HasNext(); {
		i, r := decodeHash(iter.Next(), sk.p, pp)
		sk.insert(i, r)
	}

	sk.tmpSet = nilSet
	sk.sparseList = nil
}

func (sk *Sketch) insert(i uint32, r uint8) { sk.regs[i] = max(r, sk.regs[i]) }
func (sk *Sketch) Insert(e []byte)          { sk.InsertHash(hash(e)) }

//go:noinline
func (sk *Sketch) InsertHash(x uint64) {
	if sk.sparse() {
		if sk.tmpSet.add(encodeHash(x, sk.p, pp)) {
			sk.maybeToNormal()
		}
		return
	}
	i, r := getPosVal(x, sk.p)
	sk.insert(uint32(i), r)
}

//go:noinline
func (sk *Sketch) Estimate() uint64 {
	if sk.sparse() {
		sk.mergeSparse()
		return uint64(linearCount(mp, mp-sk.sparseList.count))
	}

	sum, ez := sumAndZeros(sk.regs)
	m := float64(sk.m)

	est := sk.alpha * m * (m - ez) / (sum + beta(sk.p, ez))
	return uint64(est + 0.5)
}

func (sk *Sketch) mergeSparse() {
	if sk.tmpSet.Len() == 0 {
		return
	}

	keys := make([]uint32, 0, sk.tmpSet.Len())
	sk.tmpSet.ForEach(func(k uint32) {
		keys = append(keys, k)
	})
	slices.Sort(keys)

	newList := newCompressedList(4*sk.tmpSet.Len() + sk.sparseList.Len())
	for iter, i := sk.sparseList.Iter(), 0; iter.HasNext() || i < len(keys); {
		if !iter.HasNext() {
			newList.Append(keys[i])
			i++
			continue
		}

		if i >= len(keys) {
			newList.Append(iter.Next())
			continue
		}

		x1, adv := iter.Peek()
		x2 := keys[i]
		if x1 == x2 {
			newList.Append(x1)
			iter.Advance(x1, adv)
			i++
		} else if x1 > x2 {
			newList.Append(x2)
			i++
		} else {
			newList.Append(x1)
			iter.Advance(x1, adv)
		}
	}

	sk.sparseList = newList
	sk.tmpSet = makeSet(0)
}

// MarshalBinary implements the encoding.BinaryMarshaler interface.
//
// When the result will be appended to another buffer, consider using
// AppendBinary to avoid additional allocations and copying.
func (sk *Sketch) MarshalBinary() (data []byte, err error) {
	return sk.AppendBinary(nil)
}

// AppendBinary implements the encoding.BinaryAppender interface.
func (sk *Sketch) AppendBinary(data []byte) ([]byte, error) {
	data = slices.Grow(data, 8+len(sk.regs))
	// Marshal a version marker.
	data = append(data, version)
	// Marshal p.
	data = append(data, sk.p)
	// Marshal b
	data = append(data, 0)

	if sk.sparse() {
		// It's using the sparse Sketch.
		data = append(data, byte(1))

		// Add the tmp_set
		data, err := sk.tmpSet.AppendBinary(data)
		if err != nil {
			return nil, err
		}

		// Add the sparse Sketch
		return sk.sparseList.AppendBinary(data)
	}

	// It's using the dense Sketch.
	data = append(data, byte(0))

	// Add the dense sketch Sketch.
	sz := len(sk.regs)
	data = append(data,
		byte(sz>>24),
		byte(sz>>16),
		byte(sz>>8),
		byte(sz),
	)

	// Marshal each element in the list.
	for _, v := range sk.regs {
		data = append(data, byte(v))
	}

	return data, nil
}

// ErrorTooShort is an error that UnmarshalBinary try to parse too short
// binary.
var ErrorTooShort = errors.New("too short binary")

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface.
func (sk *Sketch) UnmarshalBinary(data []byte) error {
	if len(data) < 8 {
		return ErrorTooShort
	}

	// Unmarshal version. We may need this in the future if we make
	// non-compatible changes.
	v := data[0]

	// Unmarshal p.
	p := data[1]

	// Unmarshal b.
	b := data[2]

	// Determine if we need a sparse Sketch
	sparse := data[3] == byte(1)

	// Make a newSketch Sketch if the precision doesn't match or if the Sketch was used
	if sk.p != p || sk.regs != nil || sk.tmpSet.Len() > 0 || (sk.sparseList != nil && sk.sparseList.Len() > 0) {
		newh, err := NewSketch(p, sparse)
		if err != nil {
			return err
		}
		*sk = *newh
	}

	// h is now initialised with the correct p. We just need to fill the
	// rest of the details out.
	if sparse {
		// Using the sparse Sketch.

		// Unmarshal the tmp_set.
		tssz := binary.BigEndian.Uint32(data[4:8])
		sk.tmpSet = makeSet(int(tssz))

		// We need to unmarshal tssz values in total, and each value requires us
		// to read 4 bytes.
		tsLastByte := int((tssz * 4) + 8)
		for i := 8; i < tsLastByte; i += 4 {
			k := binary.BigEndian.Uint32(data[i : i+4])
			sk.tmpSet.add(k)
		}

		// Unmarshal the sparse Sketch.
		return sk.sparseList.UnmarshalBinary(data[tsLastByte:])
	}

	// Using the dense Sketch.
	sk.sparseList = nil
	sk.tmpSet = nilSet

	if v == 1 {
		return sk.unmarshalBinaryV1(data[8:], b)
	}
	return sk.unmarshalBinaryV2(data)
}

func sumAndZeros(regs []uint8) (res, ez float64) {
	for _, v := range regs {
		if v == 0 {
			ez++
		}
		res += 1.0 / math.Pow(2.0, float64(v))
	}
	return res, ez
}

func (sk *Sketch) unmarshalBinaryV1(data []byte, b uint8) error {
	sk.regs = make([]uint8, len(data)*2)
	for i, v := range data {
		sk.regs[i*2] = uint8((v >> 4)) + b
		sk.regs[i*2+1] = uint8((v<<4)>>4) + b
	}
	return nil
}

func (sk *Sketch) unmarshalBinaryV2(data []byte) error {
	sk.regs = data[8:]
	return nil
}

type Registers struct {
}

const (
	denseM10     = uint32(1) << 10
	denseAlpha10 = 0.7213 / (1 + 1.079/float64(denseM10))
)

type DenseSketch10 struct {
	regs [denseM10]uint8
}

//go:noinline
func (sk *DenseSketch10) InsertHash(x uint64) {
	i, r := getPosVal10(x)
	sk.regs[i] = max(r, sk.regs[i])
}

//go:noinline
func (sk *DenseSketch10) InsertHash2(x uint64) {
	// 1. Force the index to be strictly bounded within [0, 1023] at the type level
	// This helps the compiler eliminate any hidden invariant checks on fixed arrays.
	i := (x >> 54) & 1023

	// 2. Original highly optimized sentinel bit logic
	w := x<<10 | 1<<9
	rho := uint8(bits.LeadingZeros64(w)) + 1

	// 3. Localize the array lookup to break data dependency stalls
	// On M1 (ARM64), this compiles beautifully into an atomic LDR -> CSEL -> STR loop
	current := sk.regs[i]
	if rho > current {
		sk.regs[i] = rho
	}
}

func getPosVal10(x uint64) (uint64, uint8) {
	i := bextr10(x)        // {x63,...,x64-p}
	w := x<<10 | 1<<(10-1) // {x63-p,...,x0}
	rho := uint8(bits.LeadingZeros64(w)) + 1
	return i, rho
}

func bextr10(v uint64) uint64 {
	return (v >> (64 - 10)) & ((1 << 10) - 1)
}

//go:noinline
func (sk *DenseSketch10) Estimate() uint64 {
	sum, ez := sk.sumAndZeros()
	m := float64(denseM10)
	alpha := denseAlpha10
	est := alpha * m * (m - ez) / (sum + beta10(ez))
	return uint64(est + 0.5)
}

// Precompute the values once globally.
// pow2Inverse[b] stores 1.0 / (2^b) for all b in [0, 255].
// TODO: This is an easy change to make upstrem without any
// API change required.
var pow2Inverse = func() [256]float64 {
	var table [256]float64
	for i := range table {
		table[i] = 1.0 / math.Pow(2.0, float64(i))
	}
	return table
}()

func (sk *DenseSketch10) sumAndZeros() (res, ez float64) {
	// for i := 0; i < len(sk.regs); i += 8 {
	// res0 := 1.0 / math.Pow(2.0, float64(sk.regs[i]))
	// res1 := 1.0 / math.Pow(2.0, float64(sk.regs[i+1]))
	// res2 := 1.0 / math.Pow(2.0, float64(sk.regs[i+2]))
	// res3 := 1.0 / math.Pow(2.0, float64(sk.regs[i+3]))
	// res4 := 1.0 / math.Pow(2.0, float64(sk.regs[i+4]))
	// res5 := 1.0 / math.Pow(2.0, float64(sk.regs[i+5]))
	// res6 := 1.0 / math.Pow(2.0, float64(sk.regs[i+6]))
	// res7 := 1.0 / math.Pow(2.0, float64(sk.regs[i+7]))
	// res += res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7
	// }
	// return res, float64(bytes.Count(sk.regs[:], []byte{0}))
	for _, v := range sk.regs {
		if v == 0 {
			ez++
		}
		res += pow2Inverse[v]
	}
	return res, ez
}

// type sketch interface {
// 	p() uint8
// }

// TODO: Maybe we can have a generic insert function? We
// should measure if it is fast enough.
// func InsertHash[T sketch](sk T, x uint64) {
// 	i, r := getPosVal(x, sk.p())
// 	sk.insert(uint32(i), r)
// }
