# **Transcript: Hardware Support for Float-Float & Shared-Exponent Floating-Point Architecture**

**Source Document:** Float Float Hardware Support.pdf

## **Transcript**

**User:** Is there any computer architecture work providing hardware support for chaining together two 32-bit floats into close-to-64 bit double, basically an instruction set that would make implementation of float-float (like the Berkeley double-double software package) efficient on processors?  
**Assistant:** Yes. This has been studied explicitly, usually under the names double-double/float-float arithmetic, error-free transformations, twofold arithmetic, or augmented floating-point operations.  
A pair of binary32 values provides roughly 48 significant bits—not binary64's full 53—and retains binary32's exponent range. The hardware operation you most want is one that returns both the rounded result and its discarded residual:  
*a \+ b \= s \+ e*  
*a × b \= p \+ e*  
where every output is binary32.  
The most directly relevant work is:

> * **FPADDRE/FPMULRE:** Dukhan, Vuduc, and Riedy proposed FPADDRE, an instruction returning the round-off error of an addition. Thus an exact float expansion requires two independent instructions:  
>   `hi = FPADD(a, b)`  
>   `lo = FPADDRE(a, b)`They estimated substantial speedups for double-double addition, compensated dot products, and double-double matrix multiplication. The paper also proposes the analogous FPMULRE. This is almost exactly the ISA feature you describe: *Wanted: Floating-Point Add Round-off Error Instruction*.  
> * **IEEE 754-2019 augmented operations:** The standard defines recommended augmentedAddition, augmentedSubtraction, and augmentedMultiplication operations. Conceptually, these return the rounded result and residual together—the natural semantic foundation for float-float hardware. They are recommended operations, however, not mandatory processor instructions, and mainstream ISAs generally do not expose them directly. See §9.5 of IEEE 754-2019.  
> * **ReBits / Recycled Error Bits:** This architectural research preserves normally discarded rounding information in additional architectural state, allowing software to recover accuracy without fully widening every operation. It is broader than a dedicated float-float instruction but attacks the same hardware/software boundary: *Recycled Error Bits*.  
> * **Three-input exact/fused addition proposals:** An FPADD3(a,b,c) operation—adding three values with only one final rounding—can recover an addition residual using FPADD3(a, b, \-hi). It is somewhat less attractive than FPADDRE because it depends on hi and requires a more complicated adder. The FPADDRE paper discusses this alternative.

What exists in ordinary processors today is mainly an accidental partial solution:

> * **FMA makes TwoProd cheap:**  
>   `p = a * b;`  
>   `e = fma(a, b, -p);`Subject to IEEE behavior and exceptional-case qualifications, *p \+ e* reconstructs the exact product. Thus multiplication's residual costs one multiply plus one FMA. x86, Arm, RISC-V, POWER, and GPUs commonly provide FMA; for example, the standard RISC-V F extension specifies fused multiply-add operations explicitly (RISC-V F extension).

>   Addition still lacks the equivalent instruction. General TwoSum normally takes six additions/subtractions; FastTwoSum takes three but requires an operand-magnitude precondition. Consequently, addition and normalization remain the major bottlenecks.  
> * **SIMD helps run several expansions in parallel**, but it does not reduce the operation count of one expansion. Tensor-core and widening multiply-accumulate instructions are useful for mixed-precision dot products, but they generally produce a wide accumulator rather than the explicit (hi, lo) representation needed by arbitrary float-float arithmetic.

So the concise answer is: yes, there are highly relevant architectural proposals and even standardized operation semantics, but I'm not aware of a mainstream commercial ISA that currently implements a general binary32 add-with-residual or paired-result float-float instruction. FMA is the important deployed primitive, and FPADDRE IEEE augmented addition is the missing counterpart.  
---

**User:** Anybody who would bias the second float's exponent to recover something closer to the IEEE double range?  
**Assistant:** Not in conventional float-float arithmetic, as far as I can find. There is a fundamental reason: the second float's exponent is already needed to position the low-order significand relative to the first.  
For normalized float-float,  
*x \= xh \+ xl*, where |*xl*| ≤ ½ ulp(*xh*).  
If *xh* has exponent *E*, then *xl*'s exponent is normally around *E \- 24*. Thus its exponent field acts as the alignment information that gives you the extra \~24 precision bits. Biasing or repurposing it to extend *E* destroys that ordinary sum-of-two-floats interpretation.  
The cleaner construction is a scaled float-float:  
*x \= 2K(xh \+ xl)*  
where *xh*, *xl* are an ordinary normalized float-float pair and *K* is a separate integer scale. With an adequately wide *K*, this gives:

> * Approximately 48-bit significand precision;  
> * Essentially arbitrary or binary64-compatible exponent range;  
> * Reuse of ordinary binary32/FMA hardware after factoring out *2K*.

But it requires at least about 72 bits if stored straightforwardly: two binary32 words plus a separate exponent. Arithmetic must also propagate and align *K*, so it begins to resemble a small software floating-point format.  
A 64-bit version could instead encode something like:  
sign | 11-bit exponent | \~52-bit significand  
but that is essentially binary64. Splitting the significand into two float-like lanes no longer lets normal binary32 instructions operate on both components without decode, scaling, and repacking hardware.  
Related designs include:

> * **Block floating point/microscaling:** Values or components share an external scale. Modern AI formats use this heavily, but generally across vectors and for low precision, not to emulate binary64.  
> * **Classical IBM hexadecimal extended formats:** Multiword formats used constrained relationships between component exponents, although this extended precision rather than extending the range beyond the underlying format.  
> * **Floating-point expansions with explicit scaling:** Multiple-precision libraries sometimes scale operands temporarily to avoid overflow or underflow, but scaling is algorithmic metadata rather than a biased second-component exponent.  
> * **Posits/unums and tapered-precision formats:** Trade exponent and fraction bits dynamically, but they are different number systems rather than two binary32 values.

There is one potentially useful 64-bit hybrid encoding:  
*x \= (-1)s 2K (mh \+ 2\-24 ml)*  
with one shared 11-bit exponent *K*, one sign, and two approximately 24-bit significands. That would provide nearly binary64 range and around 48 bits of precision. It is better viewed as a shared-exponent, split-significand format, not float-float. Hardware could plausibly feed each significand into binary32 datapaths while managing *K* separately, but I don't know of a mainstream ISA—or a prominent established proposal that exposes exactly this format.  
So your idea is architecturally reasonable, but the useful form is "one wider shared exponent plus two significand chunks," rather than "bias the low float's exponent." The latter is representationally awkward; the former is a legitimate custom 64-bit format occupying the design space between binary32-pair arithmetic and binary64.  
---

**User:** So the idea would be to have custom instructions that would glue both the mantissas and exponents, and do it for 2 or more 32-bit floats. Custom instruction types to manage this would be ok.  
**Assistant:** Yes—once custom instructions are acceptable, this becomes a coherent architecture. I would describe it as *limbed floating point* or *shared-exponent floating-point expansions*, rather than ordinary float-float.  
The representation would be:  
*x \= (-1)s 2E (M0 \+ M1 2\-p \+ M2 2\-2p \+ …)*  
where:

> * *E* is one shared, widened exponent;  
> * Each *Mi* is a roughly 24-bit significand limb;  
> * The limbs are kept normalized and nonoverlapping;  
> * Two or more 32-bit registers carry the encoded value.

### **A practical two-word format**

Within 64 bits you could allocate:

| Field | Bits   |
| :---- | :---- |
| Sign | 1 |
| Exponent | 11 |
| Significand limbs | 48 |
| Tags/guard/classification | 4 |

That gives approximately:

> * binary64's exponent range;  
> * About 48 bits of precision versus binary64's 53;  
> * 64-bit storage;  
> * A significand naturally decomposable into two 24-bit pieces.

Conceptually:  
`word 0: sign | exponent-high | significand-high`  
`word 1: exponent-low/status | significand-low`  
The words would not independently be valid IEEE binary32 values. They are containers for one composite number.  
With three 32-bit words, an attractive format would have an 11- or 15-bit shared exponent and approximately 72 bits of precision. Four words could provide roughly 96 bits. This scales more naturally than double-double.

### **Useful custom instructions**

A minimal ISA extension might look like:  
`XFADD2   d0:d1, a0:a1, b0:b1`  
`XFSUB2   d0:d1, a0:a1, b0:b1`  
`XFMUL2   d0:d1, a0:a1, b0:b1`  
`XFFMA2   d0:d1, a0:a1, b0:b1, c0:c1`  
Each instruction would:

> 1. Reconstruct the shared exponents;  
> 2. Align both multi-limb significands;  
> 3. Operate on the limbs;  
> 4. Normalize the result;  
> 5. Round once to the composite format;  
> 6. Return the new exponent and limbs.

Supporting operations would matter too:

| Instruction | Function   |
| :---- | :---- |
| XFEXT2.S | Extract binary32 from composite |
| XFCVT.S.2 | Convert composite to binary32 |
| XFEXT2.D | Extract binary64 from composite |
| XFCVT.D.2 | Convert composite to binary64 |
| XFSCALE2 | Add an integer to the shared exponent |
| XFREPACK2 | Normalize / canonicalize limbs |
| XFCMP2 | Compare two composite values |
| XFCLASS2 | Classify composite value |

A particularly useful alternative is to expose lower-level primitives:

> * FMUL.HILO.S: rounded FP32 product and exact error  
> * FADD.HILO.S: rounded FP32 sum and exact error  
> * FNORM2.S: normalize two significand limbs  
> * FALIGN2.S: align limbs using a composite exponent

These instructions retain more software flexibility and also accelerate conventional float-float, compensated arithmetic, and robust geometry.

### **The architectural sweet spot**

| Approach | Hardware complexity | Performance / Flexibility   |
| :---- | :---- | :---- |
| Residual instructions such as FADD.HILO | Low | Improves scalar float-float software |
| Limb alignment and normalization instructions | Moderate | Flexible multi-limb arithmetic |
| Complete XFADD2 / XFMUL2 / XFFMA2 operations | Higher | Behave like native new format |

The middle option may be especially attractive. Multiplication of two two-limb numbers is essentially:  
*(a0 \+ a1)(b0 \+ b1) \= a0b0 \+ (a0b1 \+ a1b0) \+ …*  
Existing FP32 multipliers can calculate these partial products, while custom alignment and normalization hardware combines them. A processor with multiple FP32 pipes could therefore reuse much of its existing machinery.

### **Important design choice**

I would not give every limb an independently interpreted IEEE exponent. That creates:

> * Multiple representations of the same number;  
> * Expensive exponent comparisons;  
> * Difficult handling of cancellation;  
> * Wasted bits;  
> * Complicated NaN, infinity, and subnormal semantics.

Use one composite exponent and fixed-position significand limbs. The limbs can still be held in ordinary 32-bit registers, but the custom instructions interpret their fields differently.  
The result is essentially a configurable floating-point family:  
*FPX(n, p, e)*  
with *n* limbs, *p ≈ 24* bits per limb, and *e \= 11* or *15* exponent bits. For example:

> * **FPX2:** 64 bits, \~48-bit precision, 11-bit exponent  
> * **FPX3:** 96 bits, \~72-bit precision, 15-bit exponent  
> * **FPX4:** 128 bits, \~96-bit precision, 15-bit exponent

This would not quite be "two floats glued together"; it would be a native, shared-exponent floating-point format whose significand happens to use FP32-sized limbs. That distinction makes the representation and hardware substantially cleaner.  
---

**User:** Key would be to keep it 2-operand or 3-operand instructions. Collect exponent bits and cross-over to integer I/O for floating-point units would be ok.  
**Assistant:** Yes—that constraint points toward a fairly clean ISA design: keep ordinary 2-source/1-destination or 3-source/1-destination instruction shapes, while treating an integer/FP register pair as one composite operand.  
For two 32-bit words:  
`x = {control/exponent word, significand-limb word}`  
or, more practically:  
`FP register: two packed 24-bit significand limbs`  
`GPR: sign, exponent, class, status`  
The FP datapath handles the limb arithmetic; the integer side handles exponent arithmetic, classification, shifts, and exceptional cases.

### **Two-operand arithmetic**

Using destructive two-operand notation:  
`xfadd2 A, B    # A ← A + B`  
`xfmul2 A, B    # A ← A × B`  
`xfsub2 A, B    # A ← A - B`  
Each architectural operand names a register group:  
`A = {ea, a_hi, a_lo}`  
`B = {eb, b_hi, b_lo}`  
The decoder expands that into multiple physical register reads. This is similar to how architectures already handle register pairs, vector register groups, and even/odd register constraints.  
A three-operand version is preferable where encoding permits it:  
`xfadd2 D, A, B`  
`xfmul2 D, A, B`  
The arithmetic instruction still has the conventional D, A, B form; each register designator implicitly identifies a grouped composite register.

### **Separate exponent and significand instructions**

If integer/FP crossover is acceptable, a modular design may be better:  
`xfalign2  fd, fa, fb, ra, rb`  
`xfnorm2   rd, fd`  
Conceptually:

> 1. Integer registers ra and rb contain sign/exponent/class information.  
> 2. xfalign2 uses their exponent difference to align the FP limbs.  
> 3. FP adders or multipliers operate on the aligned limbs.  
> 4. xfnorm2 normalizes the result and writes the updated exponent to rd.

But crossing the integer/FP boundary for every arithmetic operation may create latency and scheduling problems. A better compromise is to keep the exponent architecturally visible in a GPR while providing complete instructions that read and write both register classes:  
`xfadd2 rd, fd, ra, fa, rb, fb`  
That is semantically clean but has too many explicit operands for most fixed-width instruction encodings.

### **Implicit register grouping**

A better encoding is:  
`xfadd2 xd, xa, xb`  
where each register number implicitly maps to:  
`xN.exponent    → integer register rN`  
`xN.significand → FP/vector register fN`  
Thus it remains a conventional three-register instruction while the microarchitecture sees:  
`reads:  rA, fA, rB, fB`  
`writes: rD, fD`  
This does increase register-file port pressure, but register renaming can internally treat each composite value as a paired physical register.  
Alternatively, use consecutive FP registers xN \= {fN, fN+1} with one word containing exponent/control information. That avoids cross-file reads but makes integer inspection less convenient. Transfer instructions can provide integer access:  
`xfgetexp rd, xa`  
`xfsetexp xd, xa, rs`  
`xfscale  xd, xa, rs`

### **A useful packed representation**

A 64-bit composite value could be divided internally as:  
`Bit 63:       sign + exponent`  
`Bits 52-0:    48-bit significand`  
The execution unit splits the 48-bit significand into two 24-bit limbs:  
*M \= Mhi 2\-24 \+ Mlo 2\-48*  
The architectural storage does not need to resemble two binary32 encodings. The internal multiplier can nevertheless reuse 24×24-bit FP32 significand multipliers.  
**This observation is important:** reuse the width and circuitry of FP32 units, not necessarily the binary32 encoding.

### **Two-result instructions**

If you want software-controlled expansion arithmetic, an instruction returning two components would require two architectural destinations:  
`faddx.s fd_hi:fd_lo, fa, fb`  
`fmulx.s fd_hi:fd_lo, fa, fb`  
That complicates instruction encoding and register renaming. There are three conventional solutions:

> * Implicit consecutive destination registers, fd and fd+1.  
> * One 64-bit vector/FP destination containing both results.  
> * A rounded result in fd and residual in a special accumulator register.

The packed 64-bit destination is probably cleanest:  
`faddx.s vd, fa, fb    # vd = {rounded sum, residual}`  
`fmulx.s vd, fa, fb    # vd = {rounded prod, residual}`  
A following composite instruction consumes that packed operand directly.

### **Recommended architecture**

I would use:

> * One 64-bit composite register operand;  
> * One shared 11-bit exponent;  
> * A 48-bit two-limb significand;  
> * Standard three-register D, A, B instruction shapes;  
> * Optional D, A, B, C FMA only if the ISA already has a four-register encoding;  
> * Integer conversion/extraction instructions at the boundary;  
> * No integer/FP crossover inside common arithmetic sequences.

The core instruction set could be as small as:  
`xadd     xd, xa, xb`  
`xsub     xd, xa, xb`  
`xmul     xd, xa, xb`  
`xfma     xd, xa, xb, xc    # optional`  
`xscale   xd, xa, rs`  
`xcmp     rd, xa, xb`  
`xcvt.s   fd, xa`  
`xcvt.x   xd, fs`  
Internally, xadd is an exponent compare, a bounded significand alignment, a 48-bit add, leading-zero detection, normalization, and packing. xmul uses four 24×24 partial products and normalization. Architecturally, though, both remain ordinary three-register operations.  
That gives you something much easier for compilers, register renaming, and out-of-order scheduling than instructions exposing every float limb separately. It is effectively a slightly lower-precision binary64 implementation constructed around reusable FP32 significand hardware.  
---

**User:** The point would be to insert this into a 16-bit accelerator architecture that has 32-bit float but not 64-bit float.  
**Assistant:** That changes the tradeoff considerably. For a 16-bit accelerator ISA with binary32 hardware but no binary64, this could be a practical way to obtain a "near-double" type without installing a complete binary64 FPU.  
The strongest design is probably:

> * A composite number occupies two 32-bit registers;  
> * Instructions name register pairs using ordinary 16-bit, two- or three-operand encodings;  
> * The format has one 11-bit exponent and a two-limb, approximately 48-bit significand;  
> * Operations reuse the accelerator's existing 24-bit FP multiplier and adder over several cycles.

### **Register-pair representation**

Use aligned register pairs:  
`x0 = {r0, r1}`  
`x1 = {r2, r3}`  
`x2 = {r4, r5}`  
The encoded register number identifies only the first register. Its low bit is implicitly zero, or the encoding names one of half as many pair registers.  
A 64-bit format might contain:

> * **Sign:** 1 bit  
> * **Exponent:** 11 bits  
> * **Significand:** 48 bits  
> * **Metadata:** 4 bits

Those four spare bits could encode:

> * Zero, infinity, and NaN classes;  
> * Sticky information;  
> * A rounding-mode override;  
> * Additional guard bits.

This gives binary64-like range and approximately 48-bit precision, five bits short of binary64.  
If IEEE binary64 interchange matters, conversion instructions can round between this format and actual 64-bit binary encodings stored in memory.

### **Fits naturally into a 16-bit ISA**

With 16 architectural pair registers, a three-address instruction can use:

| Bits 15-12 | Bits 11-8 | Bits 7-4 | Bits 3-0   |
| :---- | :---- | :---- | :---- |
| Opcode | xd | xa | xb |

For example:  
`xadd x3, x1, x2`  
`xmul x4, x3, x0`  
That gives only 16 primary opcodes, so an escape opcode or instruction prefix would probably select the extended-floating-point family.  
If the accelerator has 32 scalar registers, two-address instructions fit more comfortably:

| Bits 15-10 | Bits 9-5 | Bits 4-0   |
| :---- | :---- | :---- |
| Opcode | xd | xs |

`xadd x1, x2    # x1 = x1 + x2`  
`xmul x1, x3    # x1 = x1 * x3`  
For a small accelerator, destructive two-address operations may be perfectly reasonable, particularly if workloads are compiler-generated.

### **Reusing the FP32 hardware**

Multiplication is where this approach is most compelling. Let the composite significands be:  
*A \= a0 \+ a1 2\-24*  
*B \= b0 \+ b1 2\-24*  
Then:  
*AB \= a0b0 \+ (a0b1 \+ a1b0) 2\-24 \+ a1b1 2\-48*  
The existing 24×24-bit multiplier can calculate the four partial products sequentially:

> * **Cycle 0:** a0 × b0  
> * **Cycle 1:** a0 × b1  
> * **Cycle 2:** a1 × b0  
> * **Cycle 3:** a1 × b1  
> * Then: accumulate, normalize, and round

If only 48 output bits are required, parts of the low-low product may be reduced to sticky information. A multiply could therefore take perhaps four to seven cycles without requiring a 53×53-bit binary64 multiplier.  
Addition requires:

> 1. Compare the 11-bit exponents;  
> 2. Right-shift the smaller 48-bit significand;  
> 3. Add or subtract the limbs;  
> 4. Count leading zeros;  
> 5. Renormalize;  
> 6. Round and repack.

The wide alignment shifter is potentially more expensive than the adder. For a compact implementation it could shift 16 or 24 bits per cycle, making large exponent differences multi-cycle. If the difference is at least the significand width plus guard bits, the small operand contributes only sticky information, so it can finish quickly.

### **Suggested instruction set**

A useful minimum would be:  
`xadd     xd, xa, xb`  
`xsub     xd, xa, xb`  
`xmul     xd, xa, xb`  
`xneg     xd, xa`  
`xabs     xd, xa`  
`xcmp     flags, xa, xb`  
`xcvt.s.x fd, xa`  
`xcvt.x.s xd, fa`  
`xld      xd, [ra + imm]`  
`xst      xd, [ra + imm]`  
Then add:  
`xscale   xd, xa, ri    # exponent += i`  
`xgetexp  rd, xa`  
`xsetexp  xd, xa, ri`  
xscale would be particularly valuable in numerical kernels because it changes the composite exponent without touching the significand.  
FMA can be deferred:  
`xfma xd, xa, xb, xc`  
A four-operand instruction is awkward in 16 bits. Possible alternatives are:  
`xmac xd, xa, xb        # xd = xd + xa * xb`  
or an implicit accumulator:  
`xmac xa, xb            # xacc = xacc + xa * xb`  
The destructive three-operand xmac form is probably the best fit for an accelerator.

### **Preserve binary32 as a separate mode**

I would not encode the composite value as two independently valid floats. Treat the two words as one new format:

> * **binary32 instructions:** operate on individual words  
> * **X-format instructions:** operate on aligned register pairs

This avoids repeatedly extracting, biasing, and reconstructing two IEEE encodings. The existing FP32 datapath is reused internally at the significand level, where reuse is valuable.  
A helpful conversion instruction could nevertheless generate an ordinary float-float pair:  
`xsplit.s fhi:flo, xa`  
`xjoin.s  xd, fhi:flo`  
That permits interoperability with existing compensated algorithms.

### **Likely design point**

For a small accelerator, I would target:

| Property | Choice   |
| :---- | :---- |
| Storage | Two 32-bit registers |
| Range | 11-bit exponent, binary64-like |
| Precision | 48–51 bits, depending on metadata |
| ISA style | Three-address arithmetic or destructive MAC |
| Execution | Multi-cycle reuse of FP32 units |
| Exceptions | Simplified accelerator semantics |
| Memory | Packed 64-bit load/store, or two 32-bit transfers |

If you drop full IEEE NaN payloads, signaling NaNs, every rounding mode, and gradual underflow, you may recover more significand bits and save substantial hardware. For example, a finite-number-oriented format could use one sign bit, an 11-bit exponent, and a 52-bit significand—essentially binary64's useful numerical payload—while handling exceptional values through one reserved exponent pattern.  
At that point it is storage-compatible in width and close in semantics to binary64, but its arithmetic unit is a multi-cycle FP32-limb engine. That seems like the most convincing formulation of the idea for a 16-bit accelerator.  
---

**User:** Maximizing mantissa while having at least float64 many exponent bits would be good. I'd want a few NaNs like \+/- infinity and actual NaN/div by zero, so very little encoding space for that needed. There could be extra small registers like AVX512 mask registers for extra info related to rounding etc.  
**Assistant:** In that case, the natural 64-bit storage format converges almost exactly on binary64:

> * 1 sign bit  
> * 11 exponent bits  
> * 52 stored fraction bits  
> * 1 implicit leading significand bit

That gives 53-bit precision and binary64's exponent range. You can simplify the exceptional behavior while retaining the binary64 bit layout. The main novelty would be the implementation: custom multi-cycle instructions synthesize 64-bit arithmetic from the accelerator's existing FP32-sized datapaths.

### **Recommended encoding**

Use the standard binary64 field division:  
`Bit 63:      Sign`  
`Bits 62-52:  11-bit exponent`  
`Bits 51-0:   52-bit fraction`  
Reserve exponent \= 0x7FF for:

> * Fraction zero: positive or negative infinity;  
> * Fraction nonzero: canonical NaN.

You do not need multiple NaN encodings unless preserving diagnostic payloads matters. Inputs with any nonzero NaN fraction could canonicalize to one selected quiet-NaN representation.  
Division by zero should be an exception flag, not a number encoding. Its numerical result is normally signed infinity.  
This preserves:

> * 53 bits of significand precision;  
> * 11 exponent bits;  
> * Positive and negative infinity;  
> * NaN;  
> * Signed zero;  
> * Straightforward binary64 memory interchange.

Using the established binary64 encoding is valuable even if arithmetic is not fully IEEE compliant: debuggers, compilers, data files, and host processors can interpret stored values directly.

### **Small control and status registers**

AVX-512-style auxiliary registers are a good fit. I would separate three concepts.

#### **Floating-point control register (XRCTRL):**

| Bits | Function   |
| :---- | :---- |
| bits 1:0 | Rounding mode (00 nearest/even, 01 toward zero, etc.) |
| bit 2 | Flush subnormal inputs |
| bit 3 | Flush subnormal outputs |
| bit 4 | Saturate overflow instead of infinity |
| bit 5 | Canonicalize NaN |

If minimal hardware is important, supporting only nearest-even initially is reasonable.

#### **Sticky exception register (XRSTAT):**

`bit 0: invalid`  
`bit 1: divide by zero`  
`bit 2: overflow`  
`bit 3: underflow`  
`bit 4: inexact`  
Instructions OR new exceptions into this register. Provide ordinary operations to read, clear, and test it:  
`xgetstat  rd`  
`xclrstat`  
`xteststat rd, mask`

#### **Per-lane exception masks:**

If the accelerator is vector-like, mask registers can record which lanes produced each condition:  
`KINV   invalid lanes`  
`KDIVO  divide-by-zero lanes`  
`KOVF   overflow lanes`  
`KUNF   underflow lanes`  
`KINX   inexact lanes`  
Alternatively, one instruction selects which exception is written to a destination mask:  
`xgetexc.k k1, DIVZERO`  
`xgetexc.k k2, OVERFLOW`  
That avoids dedicating five permanent mask registers.

### **Reusing the FP32 significand hardware**

A binary64 significand has 53 bits, so two 24-bit FP32 significands are insufficient by five bits. Since your instructions reinterpret the entire 64-bit operand, there is no need to restrict the internal representation to exactly two FP32 mantissas.  
Split the 53-bit significand internally as:  
24 \+ 24 \+ 5 bits  
or more symmetrically:  
18 \+ 18 \+ 17 bits  
The 24+24+5 decomposition is best for reusing an existing 24×24 multiplier. For multiplication:  
*A \= a0 229 \+ a1 25 \+ a2*  
*B \= b0 229 \+ b1 25 \+ b2*  
The existing multiplier computes the limb partial products over multiple cycles. The implementation accumulates them into an internal product register of approximately 108 bits, then performs one final normalization and rounding.  
Crucially, the intermediate accumulator need not be architecturally visible. It can be a fixed-width integer carry-save register adjacent to the FP32 multiplier.

### **Instruction interface**

Expose each 64-bit number as an aligned pair of 32-bit registers:  
`x0 = {r0, r1}`  
`x1 = {r2, r3}`  
Then retain conventional instructions:  
`xadd x0, x1, x2`  
`xsub x0, x1, x2`  
`xmul x0, x1, x2`  
For multiply-accumulate on a constrained 16-bit encoding:  
`xmac x0, x1, x2    # x0 = round(x0 + x1 * x2)`  
That provides fused semantics without requiring four explicitly encoded registers.  
Conversions:  
`xcvt.x.s x0, f1`  
`xcvt.s.x f1, x0`  
`xcvt.x.i x0, r1`  
`xcvt.i.x r1, x0`  
Memory transfers can simply use two 32-bit accesses if the memory interface is narrow:  
`xld x0, [r1]`  
`xst x0, [r1]`  
Internally those can issue two word transactions.

### **Where simplification saves hardware**

You can retain the full 53-bit significand while simplifying expensive edge behavior:

> * Canonicalize all NaNs; do not preserve NaN payloads; omit signaling NaNs;  
> * Use sticky flags instead of traps;  
> * Optionally flush subnormal results to signed zero;  
> * Optionally treat subnormal inputs as zero;  
> * Implement division and square root iteratively;  
> * Support only nearest-even in early implementations.

Subnormal support is the significant decision. Removing gradual underflow saves normalization and shifting logic, but means the arithmetic is not fully IEEE binary64 even though normal values and special encodings remain compatible.

### **One caution about external metadata**

Rounding and exception state belongs nicely in auxiliary registers. Information necessary to reconstruct the numerical value does not. If exponent extensions or significand bits live only in mask-like registers:

> * Context switches must preserve them;  
> * Loads and stores need special handling;  
> * Register spills become complicated;  
> * Values cannot be copied as ordinary 64-bit data;  
> * Aliases between integer and FP operations become hazardous.

Therefore, keep all 64 numerical bits in the paired data registers. Use auxiliary registers only for control, sticky status, and lane masks.  
The resulting design is effectively a binary64 storage and ISA abstraction backed by a multi-cycle, FP32-derived execution engine. That achieves more precision than float-float, preserves the desired exponent range, keeps ordinary two/three-operand instructions, and avoids committing silicon area to a fully parallel binary64 FPU.  
---

**User:** Is there any work of that kind you can find?  
**Assistant:** I found several close precedents, but not an exact published architecture combining all of your choices:

> * A 16-bit instruction encoding;  
> * 32-bit scalar registers/FPU;  
> * 64-bit values held in register pairs;  
> * binary64 exponent range and nearly/full binary64 significand;  
> * Multi-cycle reuse of the FP32 datapath;  
> * Small auxiliary rounding/exception registers.

The closest work falls into four groups:

### **1\. HardFloat: arbitrary exponent and significand widths**

John Hauser's Berkeley HardFloat provides parameterized IEEE-style floating-point hardware. Its operators accept configurable expWidth and sigWidth, and it uses an internal recoded representation that makes subnormals and exceptional values easier to process.  
This is relevant because you could instantiate expWidth \= 11 and sigWidth \= 53 and connect it to paired 32-bit architectural registers. HardFloat also separates raw/recoded floating-point fields from their memory encoding, much like your proposed exponent/mantissa collection step.  
What it does not provide is the ISA or a design that sequentially decomposes 53-bit arithmetic across an existing 24-bit FP32 unit. It generally generates a datapath of the requested width. (See HardFloat documentation.)

### **2\. FPnew: multi-format FPU for small RISC-V systems**

FPnew is an open-source transprecision FPU designed for configurable processors and accelerators. It supports multiple exponent/significand formats, scalar and packed SIMD operations, configurable pipeline depths, shared execution units, format conversion, and standard rounding and exception handling.  
It is probably the closest reusable architectural framework. FPnew explicitly separates the ISA-facing format from internal functional units and allows implementations to include or omit binary64.  
However, its binary64 support is still essentially a binary64-capable datapath configuration; it does not encode one binary64 value as two FP32 limbs and replay an FP32 multiplier over them. (See FPnew paper.)

### **3\. Multi-component arithmetic on narrower units**

Henry, Tang, and Vale demonstrated the same underlying decomposition principle in the opposite precision regime: represent an FP32 value as three bfloat16 components and use narrow tensor hardware to reconstruct higher-precision products:  
*X \= X0 \+ X1 \+ X2*  
followed by multiple narrow products and a wider accumulation. This is strong precedent for using multiple narrow significand passes to emulate a wider floating-point operation. Their target was FP32-quality computation from BF16 tensor units, rather than binary64 from FP32 units. (See *Leveraging the bfloat16 AI Datatype for Higher-Precision Computation*.)  
More recent work on multi-component arithmetic similarly uses several native floating-point components plus FMA/SIMD, but retains the original component exponent range rather than assembling a wider exponent field.

### **4\. FPADDRE and augmented arithmetic**

The earlier FPADDRE proposal is the closest ISA-level precedent. It adds an instruction returning the discarded error from an addition, greatly reducing the cost of double-double arithmetic. Its proposed FPMULRE counterpart returns the product residual. (See FPADDRE paper.)  
This supports paired significands efficiently, but leaves every component in standard binary32 or binary64 format, so it does not solve your exponent-range issue.  
IEEE 754-2019's augmented addition and multiplication operations provide similar paired-result semantics but do not prescribe your composite representation.

### **What appears to be novel**

The unusual part of your proposal is not multi-cycle wide arithmetic by itself. That is common. It is the particular combination:  
`Architectural storage:  two 32-bit registers → standard 64-bit binary encoding`  
`Execution:              collect 11-bit exponent, split 53-bit significand into FP32-sized limbs,`  
                        `replay a 24-bit multiplier/adder, normalize and round once`  
`Status:                 separate mask-like rounding/exception registers`  
I have not found a documented ISA that exposes exactly that organization. The nearest established description would be: *A paired-register binary64 extension implemented by iterative decomposition over an existing binary32 significand datapath.*  
That is meaningfully different from float-float. It presents binary64—or a binary64-like format—architecturally while implementing it internally as limb arithmetic.

### **Likely research starting points**

| Work | Shared idea | Missing piece   |
| :---- | :---- | :---- |
| HardFloat | Parameterized exponent/significand and recoding | Does not iteratively reuse an FP32 datapath |
| FPnew | Configurable accelerator FPU and shared units | Binary64 generally remains a dedicated hardware format |
| BF16 decomposition | Wider precision from repeated narrow operations | Targets FP32 from BF16; keeps separate component exponents |
| FPADDRE | ISA support for multi-component arithmetic | Does not widen exponent range |
| FloPoCo | Generates arbitrary-width FP operators | FPGA generator, not a paired-register ISA |

So, based on what I can locate, there is solid prior art for every individual mechanism, but the complete architecture you describe appears underexplored and possibly novel—especially the paired-register ISA and deliberate FP32-datapath replay for a full-range, maximum-significand 64-bit format. A serious literature or patent search would still be necessary before making a formal novelty claim.  
---

**User:** But none of the current 16-bit or 32-bit AI accelerators do anything like what I am talking about?  
**Assistant:** Based on publicly documented architectures: no current AI accelerator appears to implement the complete scheme you describe—a paired-register, binary64-range value assembled from two or more 32-bit pieces and operated on by dedicated two-/three-operand instructions using repeated narrow datapath passes.  
The closest commercial example I found is Tenstorrent Tensix, and it is surprisingly close at the microarchitectural level.

### **Tenstorrent "Math Fidelity"**

Tensix splits mantissas and performs a higher-accuracy multiplication in as many as four phases:

> 1. high mantissa × high mantissa  
> 2. low mantissa × high mantissa  
> 3. high mantissa × low mantissa  
> 4. low mantissa × low mantissa

Its documentation explicitly describes HiFi2, HiFi3, and HiFi4 modes as consuming successive mantissa pieces over multiple multiplication phases. That strongly resembles the partial-product engine we discussed. (See Tenstorrent data formats and Math Fidelity.)  
But it stops short of your proposal:

> * The result remains FP32/VFP32 or an accumulator format;  
> * The exponent remains eight bits;  
> * It does not create an 11-bit composite exponent;  
> * There is no public paired-register near-binary64 format;  
> * The extra phases are selected through a fidelity mode, not exposed as general composite-scalar arithmetic instructions.

So Tensix demonstrates that replaying a narrow mantissa datapath is commercially viable, but it applies that mechanism to improving FP32 matrix accuracy rather than constructing scalar FP64.

### **What other accelerators do**

| Architecture family | Higher-precision strategy   |
| :---- | :---- |
| NVIDIA GPUs | Native FP64 on HPC models; TF32/FP16/BF16 tensor operations; software component decomposition where useful |
| AMD CDNA | Native FP64 vector/matrix hardware on HPC products |
| Google TPU | BF16/FP16 inputs with FP32 accumulation |
| Intel AMX | BF16/FP16/FP8/integer matrix products with wider accumulation |
| Tenstorrent Tensix | Multi-phase mantissa execution with FP32/VFP32 products output |
| Graphcore IPU | FP16/FP32 plus stochastic-rounding facilities |
| Typical RISC-V AI cores | FP16/BF16/FP32 via configurable units such as FPnew |

Accelerators with meaningful HPC markets—NVIDIA and AMD—usually install native FP64 hardware. Pure AI accelerators generally omit FP64 because training and inference benefit more from many narrow multipliers and FP32 accumulation.  
That leaves little commercial pressure for the middle point you identified: FP64-compatible range and near/full FP64 precision, but implemented slowly through an existing FP32 engine.

### **Software decomposition is common**

NVIDIA and other systems do use decompositions such as *X \= X0 \+ X1 \+ X2* to emulate a higher-precision product from BF16, TF32, or FP16 matrix operations. The Henry-Tang-Vale work reconstructing FP32-quality arithmetic from three BF16 pieces is a prominent example. But these are library algorithms over existing tensor instructions, not a new architectural scalar type with a composite exponent.  
Similarly, double-single software can run on FP32 accelerators, but it gets approximately 48-bit precision with only the FP32 exponent range.

### **Why your point remains distinct**

Your proposed accelerator would combine three techniques that existing systems keep separate:

> 1. Tenstorrent-like multi-phase mantissa execution  
> 2. A collected 11-bit exponent and 53-bit significand  
> 3. ISA-visible paired-register arithmetic

For example:  
`xd = {r2n, r2n+1}    # one binary64-encoded value`  
`xadd xd, xa, xb`  
`xmul xd, xa, xb`  
`xmac xd, xa, xb      # xd ← xd + xa * xb`  
Internally:

> 1. Unpack two words  
> 2. Extract 11-bit exponent  
> 3. Split 53-bit significand  
> 4. Issue several FP32-width operations  
> 5. Normalize and round once  
> 6. Repack two words

I have not found that combination in a shipping, publicly documented AI accelerator.  
The most defensible conclusion is therefore: **Tenstorrent implements the closest known execution technique, but no publicly documented current accelerator extends it into a paired-register, binary64-range scalar format.** Proprietary accelerator instruction sets and undocumented microcode make it impossible to categorically exclude every chip, but the feature is absent from the public programmer-visible formats and ISAs I could locate.
