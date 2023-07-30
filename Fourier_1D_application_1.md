---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Applications of 1D Discrete Fourier Transform

```{code-cell} ipython3
# %matplotlib inline
import numpy as np
import scipy as sc
from scipy import signal
from matplotlib import pyplot as plt
```

## Defining a test function 

To demonstrate some possible applications of the 1D Discrete Fourier transform a test function $g(t)$ is defined. The test function is first defined as a continuous function from which a discrete version will be derived.

Properties of the test function:

1. Superposition of several harmonic functions

2. Duration of test function shall be an integer of the duration a the harmonic with the lowest frequency

3. A higly oversampled discrete version of test function $g(t)$ is used

---

The continous function is composed of a DC component $A_0$ and three complex harmonics with amplitudes $A_1$, $A_2$, $A_3$ and frequencies $f_1$, $f_2$, $f_3$.

$$
g(t) = A_0 + A_1 \cdot \exp\left(j \cdot 2\pi \cdot f_1 \cdot t\right) + A_2 \cdot \exp\left(j \cdot 2\pi \cdot f_2 \cdot t\right) + A_3 \cdot \exp\left(j \cdot 2\pi \cdot f_3 \cdot t\right) 
$$

Frequencies $f_1$, $f_2$, $f_3$ shall be related to a fundamental frequency $f_s$ (sometimes referred to a frequency increment) by integer multiples $m_1$, $m_2$, $m_3$.

$$
f_1 = m_1 \cdot f_s\\
f_2 = m_2 \cdot f_s\\
f_3 = m_3 \cdot f_s\\
$$

$$
g(t) = A_0 + A_1 \cdot \exp\left(j \cdot 2\pi \cdot m_1 \cdot f_s \cdot t\right) + A_2 \cdot \exp\left(j \cdot 2\pi \cdot m_2 \cdot f_s \cdot t\right) + A_3 \cdot \exp\left(j \cdot 2\pi \cdot m_3 \cdot f_s \cdot t\right) 
$$

Function $g(t)$ is periodc with period $T_p=\frac{1}{f_s}$. The continous function $g(t)$ is converted into a time discrete  version by evaluating/sampling $g(t)$ on timing instants $k \cdot t_s$. 

$t_s$ denotes the sampling duration. $k$ are integer values running from $0, ..., N-1$. 

A period $T_p$ of $g(t)$ consists of $N$ samples. Hence the sampling duration $t_s$ is therfore $t_s = \frac{T_p}{N}$ .

With these definition we get an equation for the time discrete function $g(k \cdot t_s)$:

$$
g(k \cdot t_s) = A_0 + A_1 \cdot \exp\left(j \cdot 2\pi \cdot m_1 \cdot f_s \cdot k \cdot t_s\right) + A_2 \cdot \exp\left(j \cdot 2\pi \cdot m_2 \cdot f_s \cdot k \cdot t_s\right) + A_3 \cdot \exp\left(j \cdot 2\pi \cdot m_3 \cdot f_s \cdot k \cdot t_s\right) 
$$

Using $f_s \cdot t_s = \frac{1}{N}$:

$$
g(k \cdot t_s) = A_0 + A_1 \cdot \exp\left(j \cdot \frac{2\pi}{N} \cdot m_1 \cdot k \right) + A_2 \cdot \exp\left(j \cdot \frac{2\pi}{N} \cdot m_2 \cdot k\right) + A_3 \cdot \exp\left(j \cdot \frac{2\pi}{N} \cdot m_3 \cdot k\right) 
$$

Finally the choice of $N$ depends on the highest frequency $m_3 \cdot f_s$. With an integer oversampling factor $osr \ge 2$  $N$ is choosen like this:

$$
N = osr \cdot m_3
$$

+++

## A practical example

$$
A_0 = 1.5; A_1 = 1.0; A_2 = 4.0; A_3 = 6.0\\
m_1 = 5; m_2 = 15; m_3 = 42\\
osr = 10 \to N = 420
$$

```{code-cell} ipython3
# the definition
def tstFunc(A_0, A_1, A_2, A_3, m_1, m_2, m_3, osr):
    N = osr*m_3
    kv = np.arange(0, N)
    gfunc = A_0 + A_1 * np.exp(2j*np.pi*m_1*kv/N) + A_2 * np.exp(2j*np.pi*m_2*kv/N) + A_3 * np.exp(2j*np.pi*m_3*kv/N)
    return gfunc, kv, N
```

```{code-cell} ipython3
# function evaluation
A_0 = 1.5
A_1 = 1.0
A_2 = 4.0
A_3 = 6.0
m_1 = 5
m_2 = 15
m_3 = 42
osr = 10

gfunc, kv, N = tstFunc(A_0, A_1, A_2, A_3, m_1, m_2, m_3, osr)
```

---

The graphics below shows the real- and imaginary part in two subplots. 

```{code-cell} ipython3
# graphics
fig1 = plt.figure(1, figsize=[10, 10])
ax_f11 = fig1.add_subplot(2, 1, 1)
ax_f11.plot(kv, gfunc.real, label="real")
ax_f11.legend()
ax_f11.grid(True)
ax_f11.set_xlabel('k')
ax_f11.set_title('test function')

ax_f12 = fig1.add_subplot(2, 1, 2)
ax_f12.plot(kv, gfunc.imag, color='g', label="imag")
ax_f12.legend()
ax_f12.grid(True)
ax_f12.set_xlabel('k');
```

### Applying the DFT

Function `fft` of libraries `Scipy/Numpy`implements the DFT according to this formula:

$$
A_n = \sum_{k=0}^{N-1} a_k \cdot \exp\left(-j \cdot \frac{2\pi}{N} \cdot n \cdot k\right)
$$

To get the actual amplitudes $A_0, A_1, A_2, A_1$ a correction factor of $frac{1}{N}$ must be applied.

See example below.

```{code-cell} ipython3
# note the correction factor (1/N)
Gfft = (1/N)*np.fft.fft(gfunc)
```

In the plot below only the magnitude of the DFT is displayed. Amplitudes $A_0, A_1, A_2, A_1$ are accurately reproduce. 

A subplot below displays a zoomed version to show that amplitudes occur at their correct *frequency point" $m_1, m_2, m_3$

```{code-cell} ipython3
# graphics
fig2 = plt.figure(2, figsize=[10, 10])
ax_f21 = fig2.add_subplot(2, 1, 1)
ax_f21.plot(kv, np.abs(Gfft), label="abs")
ax_f21.legend()
ax_f21.grid(True)
ax_f21.set_xlabel('k')
ax_f21.set_title('test function / DFT')

ax_f22 = fig2.add_subplot(2, 1, 2)
ax_f22.plot(kv[0:50], np.abs(Gfft[0:50]), label="abs")
ax_f22.legend()
ax_f22.grid(True)
ax_f22.set_xlabel('k')
ax_f22.set_title('zoomed display');
```

```{code-cell} ipython3

```
