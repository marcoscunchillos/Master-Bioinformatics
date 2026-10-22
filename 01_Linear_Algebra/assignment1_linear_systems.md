Linear Systems in Data Science
================
Marcos Cunchillos Just
2026-10-22

- [Overview](#overview)
- [Part 1. Least-squares regression](#part-1-least-squares-regression)
  - [Theory](#theory)
  - [Closed-form helper](#closed-form-helper)
  - [Dataset A: fit and plot](#dataset-a-fit-and-plot)
  - [Dataset B: fit, predictions and
    residuals](#dataset-b-fit-predictions-and-residuals)
- [Part 2. Steady-state temperature of a metal
  plate](#part-2-steady-state-temperature-of-a-metal-plate)
  - [Problem](#problem)
  - [Building the system](#building-the-system)
  - [Solving the system and discussing its
    type](#solving-the-system-and-discussing-its-type)
  - [Temperatures at the four
    corners](#temperatures-at-the-four-corners)
- [Conclusions](#conclusions)
- [Session information](#session-information)

# Overview

This report applies the theory of **linear systems** to two classic
problems:

| Part | Problem | Linear-algebra idea |
|----|----|----|
| 1 | Simple linear regression on two datasets | Overdetermined system solved by least squares (normal equations) |
| 2 | Steady-state temperature of a metal plate | Square, sparse, symmetric system $A\mathbf{t} = \mathbf{b}$ with a unique solution |

**Tools:** R, `ggplot2`, base `lm()` / `solve()` / `qr()`.

------------------------------------------------------------------------

# Part 1. Least-squares regression

## Theory

Given $n$ observations $(x_1, y_1), \dots, (x_n, y_n)$ of two
characteristics of a set of individuals, we fit a linear model
$y = ax + b$ that minimizes the squared distance between the model
predictions and the observed values:

$$E(a, b) = \sum_{i=1}^{n} (a x_i + b - y_i)^2 .$$

Setting the partial derivatives to zero gives a linear system in the
unknowns $a$ and $b$:

$$
\frac{\partial E}{\partial a} = 2\sum_{i=1}^{n} (a x_i + b - y_i)\,x_i = 0,
\qquad
\frac{\partial E}{\partial b} = 2\sum_{i=1}^{n} (a x_i + b - y_i) = 0 .
$$

Dividing both equations by $n$ and writing
$\bar{x} = \frac{1}{n}\sum x_i$,
$\overline{x^2} = \frac{1}{n}\sum x_i^2$,
$\overline{xy} = \frac{1}{n}\sum x_i y_i$, etc., the system becomes

$$
a\,\overline{x^2} + b\,\bar{x} = \overline{xy},
\qquad
a\,\bar{x} + b = \bar{y}.
$$

From the second equation, $b = \bar{y} - a\bar{x}$. Substituting into
the first one:

$$
a\left(\overline{x^2} - \bar{x}^2\right) = \overline{xy} - \bar{x}\bar{y}
\quad\Longrightarrow\quad
a = \frac{\overline{xy} - \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} = \frac{\mathrm{cov}(x, y)}{\mathrm{var}(x)} .
$$

The regression line therefore always passes through the centroid
$(\bar{x}, \bar{y})$.

## Closed-form helper

``` r
#' Least-squares slope and intercept from the closed-form formulas
#'
#' @param x Numeric vector of predictor values.
#' @param y Numeric vector of responses (same length as `x`).
#' @return Named numeric vector with the intercept `b` and the slope `a`.
ols_line <- function(x, y) {
  a <- sum((x - mean(x)) * (y - mean(y))) / sum((x - mean(x))^2)
  b <- mean(y) - a * mean(x)
  c(b = b, a = a)
}
```

## Dataset A: fit and plot

Eight observations $(X, Y)$:
$(-10,-15),\ (-5,1),\ (0,1),\ (5,3),\ (10,20),\ (10,7),\ (15,42),\ (20,50)$.

``` r
x_a <- c(-10, -5, 0, 5, 10, 10, 15, 20)
y_a <- c(-15,  1, 1, 3, 20,  7, 42, 50)
data_a <- data.frame(x = x_a, y = y_a)

model_a <- lm(y ~ x, data = data_a)
coef(model_a)
```

    ## (Intercept)           x 
    ##    2.350649    2.004329

``` r
# The closed-form solution must agree with lm()
all.equal(unname(ols_line(x_a, y_a)), unname(coef(model_a)))
```

    ## [1] TRUE

``` r
# Goodness of fit
summary(model_a)$r.squared
```

    ## [1] 0.8372166

The fitted line is $y \approx 2.004\,x + 2.351$.

``` r
ggplot(data_a, aes(x = x, y = y)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", colour = "red", fill = "red", alpha = 0.15) +
  labs(title = "Dataset A: data and least-squares line",
       x = "X", y = "Y") +
  theme_minimal()
```

<img src="assignment1_linear_systems_files/figure-gfm/plot-a-1.png" alt="" style="display: block; margin: auto;" />

**Does the line fit the data well?** Reasonably well:
$R^2 \approx 0.84$, so about 84 % of the variability in $Y$ is explained
by $X$, and the trend is clearly increasing. The two observations at
$x = 10$ ($y = 20$ and $y = 7$) show that the scatter around the line is
large, and the confidence band is wide, so predictions at individual
points remain imprecise.

## Dataset B: fit, predictions and residuals

Laboratory measurements of two characteristics of ten patients.

``` r
x_b <- c(4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14)
y_b <- c(4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69)
data_b <- data.frame(x = x_b, y = y_b)

model_b <- lm(y ~ x, data = data_b)
coef(model_b)
```

    ## (Intercept)           x 
    ##   7.7957139  -0.6229559

``` r
summary(model_b)$r.squared
```

    ## [1] 0.2094603

``` r
# Model predictions: y_hat_i = a * x_i + b
data_b$y_hat <- predict(model_b)
```

The fitted line is $y \approx -0.623\,x + 7.796$.

The plot overlays the observed data, the regression line and the model
predictions $\hat{y}_i = a x_i + b$. By construction, all predictions
lie on the line, and the dashed segments are the residuals
$e_i = y_i - \hat{y}_i$ that least squares minimizes.

``` r
ggplot(data_b, aes(x = x)) +
  geom_segment(aes(xend = x, y = y, yend = y_hat),
               colour = "grey60", linetype = "dashed") +
  geom_smooth(aes(y = y), method = "lm", se = FALSE,
              colour = "red", linewidth = 0.8) +
  geom_point(aes(y = y, colour = "Observed data"), size = 2) +
  geom_point(aes(y = y_hat, colour = "Model predictions"),
             shape = 17, size = 3) +
  scale_colour_manual(values = c("Observed data" = "black",
                                 "Model predictions" = "blue")) +
  labs(title = "Dataset B: observed data, fitted line and predictions",
       x = "X", y = "Y", colour = NULL) +
  theme_minimal()
```

<img src="assignment1_linear_systems_files/figure-gfm/plot-b-1.png" alt="" style="display: block; margin: auto;" />

**Interpretation.** The slope is negative, but the fit is weak
($R^2 \approx 0.21$): $X$ explains only about a fifth of the variability
in $Y$, and the points are widely scattered around the line. With only
ten observations, this association should not be considered reliable
without further data or a significance test.

------------------------------------------------------------------------

# Part 2. Steady-state temperature of a metal plate

## Problem

The temperature (in °C) is known along the perimeter of a square plate.
The unknowns are the nine interior temperatures $T_{22}, \dots, T_{44}$:

$$
T =
\begin{pmatrix}
\cdot & 95 & 87 & 63 & \cdot \\
53 & T_{22} & T_{23} & T_{24} & 51 \\
29 & T_{32} & T_{33} & T_{34} & 52 \\
25 & T_{42} & T_{43} & T_{44} & 48 \\
\cdot & 18 & 21 & 20 & \cdot
\end{pmatrix}
$$

**Model assumption:** the temperature at each interior point equals the
mean of its four neighbours (up, down, left, right). For example,

$$
T_{22} = \frac{T_{12} + T_{21} + T_{23} + T_{32}}{4}
\;\Longleftrightarrow\;
4\,T_{22} - T_{23} - T_{32} = 95 + 53 = 148 .
$$

Repeating this for the nine interior points gives a $9 \times 9$ linear
system $A\mathbf{t} = \mathbf{b}$ with
$\mathbf{t} = (T_{22}, T_{23}, T_{24}, T_{32}, T_{33}, T_{34}, T_{42}, T_{43}, T_{44})^\top$.
Each row of $A$ has a `4` on the diagonal and a `-1` for every interior
neighbour, and the known boundary temperatures move to the right-hand
side $\mathbf{b}$. This is the finite-difference discretization of the
Laplace equation (steady-state heat diffusion).

## Building the system

``` r
# Coefficient matrix, written explicitly (one row per interior point)
A <- matrix(c(
   4, -1,  0, -1,  0,  0,  0,  0,  0,
  -1,  4, -1,  0, -1,  0,  0,  0,  0,
   0, -1,  4,  0,  0, -1,  0,  0,  0,
  -1,  0,  0,  4, -1,  0, -1,  0,  0,
   0, -1,  0, -1,  4, -1,  0, -1,  0,
   0,  0, -1,  0, -1,  4,  0,  0, -1,
   0,  0,  0, -1,  0,  0,  4, -1,  0,
   0,  0,  0,  0, -1,  0, -1,  4, -1,
   0,  0,  0,  0,  0, -1,  0, -1,  4
), nrow = 9, byrow = TRUE)

# Right-hand side: sum of the known boundary neighbours of each interior point
b <- c(95 + 53,  87,  63 + 51,
       29,        0,  52,
       25 + 18,  21,  20 + 48)

# Sanity checks: the matrix must be 9 x 9 and symmetric
dim(A)
```

    ## [1] 9 9

``` r
isSymmetric(A)
```

    ## [1] TRUE

## Solving the system and discussing its type

By the Rouché-Frobenius theorem, the system has a unique solution when
\$(A) = (\[A \]) = \$ number of unknowns.

``` r
# 1. Classify the system
qr(A)$rank              # rank of A
```

    ## [1] 9

``` r
qr(cbind(A, b))$rank    # rank of the augmented matrix
```

    ## [1] 9

``` r
# 2. Solve
t_interior <- solve(A, b)
names(t_interior) <- c("T22", "T23", "T24", "T32", "T33", "T34", "T42", "T43", "T44")
round(t_interior, 2)
```

    ##   T22   T23   T24   T32   T33   T34   T42   T43   T44 
    ## 63.43 63.45 56.42 42.29 46.94 48.21 29.77 33.80 37.50

``` r
# 3. Verify the solution
all.equal(as.vector(A %*% t_interior), b)
```

    ## [1] TRUE

**Is the solution unique?** Yes.
$\mathrm{rank}(A) = \mathrm{rank}([A \mid \mathbf{b}]) = 9$, which
equals the number of unknowns, so the system is *compatible determinate*
(equivalently, $\det(A) \neq 0$ and $A$ is invertible).

**Hottest and coldest interior points.** The maximum is at
$T_{23} \approx 63.45\,^\circ\mathrm{C}$ (very close to
$T_{22} \approx 63.43\,^\circ\mathrm{C}$), next to the hot upper edge.
The minimum is at $T_{42} \approx 29.77\,^\circ\mathrm{C}$, next to the
cold lower-left boundary.

``` r
names(which.max(t_interior)); max(t_interior)
```

    ## [1] "T23"

    ## [1] 63.44643

``` r
names(which.min(t_interior)); min(t_interior)
```

    ## [1] "T42"

    ## [1] 29.77232

## Temperatures at the four corners

Each corner has only three neighbours, so we take the mean of those
three values:

| Corner   | Neighbours                             |
|----------|----------------------------------------|
| $T_{11}$ | $T_{12} = 95$, $T_{21} = 53$, $T_{22}$ |
| $T_{15}$ | $T_{14} = 63$, $T_{25} = 51$, $T_{24}$ |
| $T_{51}$ | $T_{41} = 25$, $T_{52} = 18$, $T_{42}$ |
| $T_{55}$ | $T_{45} = 48$, $T_{54} = 20$, $T_{44}$ |

``` r
corners <- c(
  T11 = (95 + 53 + t_interior[["T22"]]) / 3,
  T15 = (63 + 51 + t_interior[["T24"]]) / 3,
  T51 = (25 + 18 + t_interior[["T42"]]) / 3,
  T55 = (48 + 20 + t_interior[["T44"]]) / 3
)
round(corners, 2)
```

    ##   T11   T15   T51   T55 
    ## 70.48 56.81 24.26 35.17

------------------------------------------------------------------------

# Conclusions

- Least-squares regression reduces to a $2 \times 2$ linear system whose
  solution is $a = \mathrm{cov}(x, y)/\mathrm{var}(x)$ and
  $b = \bar{y} - a\bar{x}$; it matches `lm()` exactly.
- $R^2$ quantifies fit quality: Dataset A shows a strong linear trend
  ($R^2 \approx 0.84$), whereas Dataset B shows a weak association
  ($R^2 \approx 0.21$).
- The plate problem is a square, symmetric, sparse system with a unique
  solution; checking `rank(A)` against `rank([A | b])` before calling
  `solve()` is the practical form of the Rouché-Frobenius discussion.
- For larger grids, direct elimination becomes expensive, which
  motivates iterative methods (Jacobi, Gauss-Seidel) and sparse matrix
  storage.

# Session information

``` r
sessionInfo()
```

    ## R version 4.6.0 (2026-04-24 ucrt)
    ## Platform: x86_64-w64-mingw32/x64
    ## Running under: Windows 11 x64 (build 26200)
    ## 
    ## Matrix products: default
    ##   LAPACK version 3.12.1
    ## 
    ## locale:
    ## [1] LC_COLLATE=Spanish_Spain.utf8  LC_CTYPE=Spanish_Spain.utf8   
    ## [3] LC_MONETARY=Spanish_Spain.utf8 LC_NUMERIC=C                  
    ## [5] LC_TIME=Spanish_Spain.utf8    
    ## 
    ## time zone: Europe/Madrid
    ## tzcode source: internal
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ## [1] ggplot2_4.0.3
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] vctrs_0.7.3        nlme_3.1-169       cli_3.6.6          knitr_1.51        
    ##  [5] rlang_1.2.0        xfun_0.57          generics_0.1.4     S7_0.2.2          
    ##  [9] labeling_0.4.3     glue_1.8.1         htmltools_0.5.9    scales_1.4.0      
    ## [13] rmarkdown_2.31     grid_4.6.0         evaluate_1.0.5     tibble_3.3.1      
    ## [17] fastmap_1.2.0      yaml_2.3.12        lifecycle_1.0.5    compiler_4.6.0    
    ## [21] dplyr_1.2.1        RColorBrewer_1.1-3 pkgconfig_2.0.3    mgcv_1.9-4        
    ## [25] rstudioapi_0.18.0  lattice_0.22-9     farver_2.1.2       digest_0.6.39     
    ## [29] R6_2.6.1           tidyselect_1.2.1   splines_4.6.0      pillar_1.11.1     
    ## [33] magrittr_2.0.5     Matrix_1.7-5       withr_3.0.2        tools_4.6.0       
    ## [37] gtable_0.3.6
