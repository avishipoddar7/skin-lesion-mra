# How This Project Works

Three parts, from simplest to deepest:

1. [What happens in the app](#1-what-happens-in-the-app-simple-steps)
2. [What we are doing, and why](#2-what-we-are-doing-and-why)
3. [The theory: how each technique works and why we use it](#3-the-theory-how-each-technique-works-and-why-we-use-it)

---

## 1. What happens in the app (simple steps)

You open the app, pick an image (or one of the three samples), choose a few settings, and press
**Analyse**. Here is what the program does, in order:

| Step | What happens | Where you see it |
|---|---|---|
| 1 | **Load and check the image.** Only JPG, PNG and BMP files of a sensible size are accepted. Anything else shows an error card. | Error card |
| 2 | **Make it square and small.** The image is padded to a square (so the lesion is not stretched) and shrunk to 256×256 pixels. | "Original" picture |
| 3 | **Remove hair.** Thin dark hairs are found and painted over using the surrounding skin. | Hair mask, Hair removed |
| 4 | **Turn it grey and split it into scales.** A wavelet transform splits the picture into one blurry "big picture" (LL) and several "detail" pictures (edges and texture). | Subband pictures (LL, LH, HL, HH) |
| 5 | **Find the lesion.** On the blurry big picture, the program picks a brightness cut-off that separates the dark mole from the lighter skin, then tidies the shape. | Coarse mask, Final mask, Contour overlay |
| 6 | **Measure the lesion.** It calculates numbers: how symmetric it is, how smooth its border is, its colours, its size, and how much texture it has at each scale. | ABCD table, Subband energy chart |
| 7 | **Make a prediction.** A trained classifier (an SVM) looks at those numbers and gives a **melanoma score**. If the score is 20% or higher, the lesion is flagged as melanoma. | Prediction card |
| 8 | **Show everything, with a disclaimer.** Every step is displayed so you can see how the answer was reached. | All three tabs |

Two things worth knowing:

- **Your settings change what you see, not the prediction.** The classifier was trained with one
  fixed setup (Haar wavelet, level 2, hair removal on). If you choose db4 or level 3, the pictures
  change, but the prediction still comes from the default setup, and the app tells you so.
- **The sample images are from the training data**, so their results look better than real performance.
  The app labels them as such.

---

## 2. What we are doing, and why

### The problem
Melanoma is the most dangerous skin cancer, but it is very treatable when caught early. Doctors use
a dermatoscope (a magnifying, lit camera) and judge a mole by the **ABCD rule**: **A**symmetry,
**B**order, **C**olour, **D**iameter. Specialists are scarce, so automatic help is useful.

### Our approach
Most modern work uses deep learning, which needs lots of data and computing power and cannot explain
its answers. We do something different and explainable:

- **Classical image processing** (wavelets and morphology) instead of a neural network.
- **Every step is visible** in the app.
- **Systematic comparison:** we test 3 wavelets (Haar, db4, sym4) × 3 levels (1, 2, 3) × hair removal
  on/off = 18 configurations, to see what actually matters.

### The data
The **PH2 dataset**: 200 dermoscopy images, each with an expert-drawn outline of the lesion and a
diagnosis: 80 common nevi, 80 atypical nevi, 40 melanomas. We treat it as a two-class problem:
melanoma (40) vs. non-melanoma (160).

### How we measure success
| Question | Measure | Result (default setup) |
|---|---|---|
| Is the outline right? | **Dice** score against the expert outline | 0.81 |
| Does it catch melanomas? | **Sensitivity** (share of melanomas found) | 0.85 |
| Does it avoid false alarms? | **Specificity** (share of non-melanomas cleared) | 0.84 |
| Overall ranking quality | **AUC** | 0.92 |

Classification results use **5-fold cross-validation** (the data is split in 5 parts; each part is
tested by a model trained on the other four), repeated over 3 random splits. This avoids testing on
the images the model learned from.

### What we found
- **Haar segments best** (Dice 0.81). The longer wavelets db4 and sym4 drop to 0.65–0.75 at levels 2–3.
- **Hair removal** barely changes the outline but improves classification slightly (AUC 0.90 → 0.92).
- **ABCD features alone** classify as well as ABCD + wavelet texture together. The wavelet texture
  features add explainability but no extra accuracy here.
- **Melanomas are harder to outline** (Dice 0.62) than ordinary moles (0.86), because they are larger
  and more irregular.
- **Limits:** only 40 melanomas, so scores are noisy; PH2 is mostly fair skin; this is a student demo,
  not a medical device.

---

## 3. The theory: how each technique works and why we use it

### 3.1 Hair removal (morphology: black-hat + inpainting)

**The problem.** Hairs lie across the lesion. They are dark and thin, so they confuse the outline and
distort the colour and texture measurements.

**Mathematical morphology** processes shapes in an image using a small "probe" called a
*structuring element* (here an 11×11 ellipse). Two basic operations:

- **Dilation** grows bright regions (the probe takes the *maximum* under it).
- **Erosion** shrinks bright regions (the probe takes the *minimum*).
- **Closing** = dilation then erosion. It fills in dark features *narrower than the probe*.
- **Opening** = erosion then dilation. It removes bright specks narrower than the probe.

**The black-hat transform** = Closing(image) − image.
Closing fills thin dark lines (hairs) with the surrounding brightness. Subtracting the original
leaves *only* those thin dark features as bright pixels. The probe must be **wider than a hair but
smaller than the lesion**, otherwise the whole lesion would be "filled" and treated as hair.

**Our pipeline for hair:**
1. Black-hat on the grey image.
2. Threshold at 25: pixels above it are hair candidates. (The first guess of 10 marked about 34% of
   every image as hair, wiping out real texture, so we raised it.)
3. Drop blobs smaller than 40 pixels (noise specks).
4. Dilate slightly so the hair's edges are included.
5. **Inpainting (Telea method):** each hair pixel is refilled from its neighbours, working inwards
   from the edge of the hair, using a weighted average that favours nearby pixels and pixels along
   the local edge direction. The result is a smooth patch of plausible skin.

**Honest caveat.** Black-hat cannot tell a hair from other thin dark things, so it also touches
pigment dots and network lines inside some lesions. That is why we test hair removal on and off.

### 3.2 Vignette suppression

Dermoscopy images are taken through a circular lens, leaving a dark ring around the image. Otsu
thresholding (below) sees this ring as "dark, so lesion" and merges it with large lesions. Before
segmenting, we find dark regions that touch the image border and fill them with the median grey
level. This one step raised Dice from 0.75 to 0.81.

### 3.3 The wavelet transform (why Haar, how it works)

**Idea.** A signal contains information at several scales: the overall shape (coarse) and the sharp
edges and texture (fine). A wavelet transform separates them.

**Haar, in one dimension.** Take pairs of neighbouring values `(a, b)`:

- **Average (low-pass):** `(a + b) / √2` — the smooth part
- **Difference (high-pass):** `(a − b) / √2` — the detail part

Worked example, signal `[4, 6, 10, 12]`:

| Pair | Average | Difference |
|---|---|---|
| (4, 6) | 10/√2 = 7.07 | −2/√2 = −1.41 |
| (10, 12) | 22/√2 = 15.56 | −2/√2 = −1.41 |

The signal is now half as long (the averages `[7.07, 15.56]`) plus the details `[−1.41, −1.41]`.
Nothing is lost: the original can be rebuilt exactly from both parts.

**In two dimensions** (an image), Haar is applied along rows and columns, turning each 2×2 block of
pixels into four numbers, one in each of four sub-images (*subbands*), each half the size:

| Subband | Meaning | Looks like |
|---|---|---|
| **LL** | average of the block (low-pass both ways) | a smaller, blurrier copy of the image |
| **LH** | change between top and bottom rows | horizontal edges |
| **HL** | change between left and right columns | vertical edges |
| **HH** | change along the diagonal | diagonal edges, fine texture |

(For a 2×2 block `[[a, b], [c, d]]`, LL is `(a+b+c+d)/2`; the others are the signed differences.)

**Multilevel.** Level 2 applies the same split again to the LL band. A 256×256 image becomes 128×128
at level 1 and 64×64 at level 2. Each level looks at coarser structure. This is Mallat's
multiresolution algorithm.

**Why Haar, db4, sym4.** They are different filter shapes. Haar has the shortest filter (2 taps), so
it keeps sharp boundaries and does not smear across them. db4 and sym4 have longer, smoother filters.
Our experiments found Haar gave the best outline; the longer filters, which average over a wider area,
likely blur and shift the lesion border at coarse levels (Dice fell to 0.65 for db4 at level 3). This
is a likely explanation, not something we proved.

### 3.4 Segmentation (Otsu on the LL band, then morphology)

**Why use LL.** The LL band is an *averaged, downsampled* copy of the image. Averaging washes out
leftover hair, noise and fine pigment texture, leaving a clean "dark blob on light skin" picture.
Thresholding that is much more reliable than thresholding the raw image.

**Otsu's method** picks the brightness cut-off automatically. It tries every possible threshold and
chooses the one that best separates pixels into two groups, meaning the one that maximises the
between-group variance of brightness (equivalently, minimises the spread inside each group). Because
the lesion is *darker* than skin, we keep the dark group (an "inverse" threshold).

**Cleaning up the mask** (all morphology again):
1. Stretch the 64×64 mask back to 256×256.
2. **Opening** removes small specks.
3. **Closing** joins gaps and smooths the border.
4. **Hole filling** makes the lesion solid.
5. **Pick one region:** of the remaining blobs, keep the largest, with a penalty for ones touching the
   image border.

If the result is under 1% or over 90% of the image, the app reports "no lesion found".

**Scoring it: Dice and Jaccard.** With our mask A and the expert mask B:
- Dice = 2·|A ∩ B| / (|A| + |B|)
- Jaccard = |A ∩ B| / |A ∪ B|

Both are 1 for a perfect match and 0 for no overlap.

### 3.5 Image pyramids

A **Gaussian pyramid** repeatedly blurs the image and halves its size. A **Laplacian pyramid** stores,
at each level, the *difference* between that level and the blurred, enlarged next level, so it holds
the detail lost in one blur-and-shrink step. This is also multiresolution analysis, but pyramids are
*redundant* (they store about 4/3 as many numbers as the image), while the wavelet transform is not
(same count of numbers as the image). In this project the pyramids are for display and teaching
(Unit 6); the segmentation and features use the wavelet transform.

### 3.6 Features: turning a lesion into numbers

**Wavelet texture features** (computed on a 128×128 window around the lesion), for each level and each
detail subband (H, V, D):
- **Energy** = mean of c², where c are the coefficients. High energy means strong edges or texture at
  that scale and direction.
- **Entropy** = −Σ p·log₂(p), where p = c² / Σc². High entropy means the energy is spread evenly
  (busy, irregular texture); low means it is concentrated in a few places.
- Plus the mean and spread of the LL band.

**ABCD features** (the doctor's rule, as numbers):

| Letter | Feature | How it is calculated |
|---|---|---|
| **A**symmetry | `asymmetry` | Flip the lesion about each of its two main axes, count the area that does not overlap, divide by lesion area, average the two. 0 = perfectly symmetric. |
| **B**order | `border_ci` | Compactness = perimeter² / (4π · area). A perfect circle = 1; ragged borders give larger values. |
| | `solidity` | Lesion area / area of its convex outline. Lower means a more indented border. |
| **C**olour | `color_R/G/B_mean, _std` | Average and spread of each colour channel inside the lesion. More colours means more spread. |
| **D**iameter | `rel_diameter` | Length of the lesion's long axis as a fraction of the image width. |

### 3.7 Classification (SVM)

A **Support Vector Machine** finds a boundary that separates the two classes with the widest possible
margin. We use the **RBF kernel**, which lets the boundary curve so it can separate classes that a
straight line cannot. Choices we made, and why:

- **Standardise features** first, so numbers with big ranges (colour values) do not drown out small
  ones (solidity).
- **Balanced class weights**: melanoma is only 20% of the data, so mistakes on melanomas are weighted
  more heavily, otherwise the model would just say "non-melanoma" for everything.
- **Calibration** turns the SVM's raw output into a 0–1 score that behaves like a probability.
- **Cut-off at 0.20**, the melanoma share of the dataset. The usual 0.5 cut-off missed about 40% of
  melanomas (sensitivity 0.55–0.62); at 0.20 sensitivity is 0.85. We set it by this reasoning, not by
  searching for the best value on our results. This is a screening trade-off: more false alarms in
  exchange for fewer missed melanomas.

The score is a **model estimate**, not a clinical probability.

### 3.8 Why the results look the way they do

| Result | Likely reason |
|---|---|
| Haar > db4/sym4 for segmentation | Short filter keeps lesion edges sharp (explanation, not proven) |
| Dice lower for melanoma | Melanomas are large (some fill 90%+ of the frame), irregular and mottled, so a single brightness cut-off splits them badly |
| Hair removal helps AUC, not Dice | Hair mainly disturbs colour and texture measurements; the LL band already averages most of it away when outlining |
| ABCD alone ≈ combined | The ABCD features already capture what separates the classes in this small dataset; with only 40 melanomas, extra features add noise as well as signal |
| Lesion size (diameter) is a strong cue | In PH2, melanomas tend to be bigger, so the classifier may lean on size. A larger, more varied dataset would test this |

### 3.9 Where each idea comes from (course units)

| Topic | Unit | Where used |
|---|---|---|
| Wavelet transform, Haar, multilevel decomposition | 6 | `src/wavelet_utils.py`, segmentation, features |
| Image pyramids | 6 | `src/wavelet_utils.py`, app "Pyramids" panel |
| Morphology (black-hat, opening, closing, hole filling) | 7 | `src/preprocessing.py`, `src/segmentation.py` |
| Thresholding (Otsu) | 7 | `src/segmentation.py` |
| Medical application: melanoma detection | 7 | the whole project |

References: Garnavi et al. 2012 (wavelet + border features for melanoma); Mallat 1989
(multiresolution theory); Otsu 1979 (thresholding); Barata et al. 2019 (feature survey);
Talavera-Martínez et al. 2021 (hair removal); Yu et al. 2017 (deep-learning comparison).
