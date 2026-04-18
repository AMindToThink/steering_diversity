Sure. This is somewhat involved, but the result simplifies nicely when you decompose $\delta$ into components parallel and perpendicular to $\hat{m}$.

## The Hessian of the sphere projection

We need $\frac{\partial^2 g_i}{\partial v_j \partial v_k}$. Starting from the Jacobian entry we already derived:

$$\frac{\partial g_i}{\partial v_j} = \frac{\delta_{ij}}{\|v\|} - \frac{v_i v_j}{\|v\|^3}$$

Differentiate with respect to $v_k$:

$$\frac{\partial^2 g_i}{\partial v_j \partial v_k} = -\frac{\delta_{ij} v_k}{\|v\|^3} - \frac{\delta_{ik} v_j + v_i \delta_{jk}}{\|v\|^3} + \frac{3 v_i v_j v_k}{\|v\|^5}$$

The first term comes from differentiating $1/\|v\|$ in the first piece. The middle terms come from differentiating $v_i v_j$ in the numerator of the second piece. The last term comes from differentiating $1/\|v\|^3$ via chain rule.

Evaluated at $v = m$ and contracting with $\delta_j \delta_k$ (i.e., computing the $i$-th component of $H[\delta, \delta]$):

- $-\delta_{ij} m_k \delta_j \delta_k / \|m\|^3 = -\delta_i (\hat{m}^\top \delta) / \|m\|^2$
- $-\delta_{ik} m_j \delta_j \delta_k / \|m\|^3 = -(\hat{m}^\top \delta) \delta_i / \|m\|^2$
- $-m_i \delta_{jk} \delta_j \delta_k / \|m\|^3 = -\hat{m}_i \|\delta\|^2 / \|m\|^2$
- $3 m_i m_j m_k \delta_j \delta_k / \|m\|^5 = 3 \hat{m}_i (\hat{m}^\top \delta)^2 / \|m\|^2$

In vector form:

$$H[\delta, \delta] = \frac{1}{\|m\|^2}\left[-2(\hat{m}^\top \delta)\,\delta - \|\delta\|^2\,\hat{m} + 3(\hat{m}^\top \delta)^2\,\hat{m}\right]$$

## Decomposing into radial and perpendicular

Let $\alpha = \hat{m}^\top \delta$ (radial component) and $\delta_\perp = P_{\perp m}\,\delta$ (perpendicular component), so $\delta = \alpha\,\hat{m} + \delta_\perp$ and $\|\delta\|^2 = \alpha^2 + \|\delta_\perp\|^2$.

Substitute into $H[\delta,\delta]$:

$$H[\delta,\delta] = \frac{1}{\|m\|^2}\left[-2\alpha(\alpha\,\hat{m} + \delta_\perp) - (\alpha^2 + \|\delta_\perp\|^2)\,\hat{m} + 3\alpha^2\,\hat{m}\right]$$

Collecting the $\hat{m}$ and $\delta_\perp$ components:

$$= \frac{1}{\|m\|^2}\left[(-2\alpha^2 - \alpha^2 - \|\delta_\perp\|^2 + 3\alpha^2)\,\hat{m} - 2\alpha\,\delta_\perp\right]$$

$$= \frac{1}{\|m\|^2}\left[-\|\delta_\perp\|^2\,\hat{m} - 2\alpha\,\delta_\perp\right]$$

The $\alpha^2$ terms cancel exactly. This is the nice simplification.

## The full second-order expansion

$$z \approx \hat{m} + \frac{1}{\|m\|}\delta_\perp + \frac{1}{2\|m\|^2}\left[-\|\delta_\perp\|^2\,\hat{m} - 2\alpha\,\delta_\perp\right]$$

Regrouping by direction:

$$z \approx \underbrace{\left(1 - \frac{\|\delta_\perp\|^2}{2\|m\|^2}\right)}_{\text{radial component}}\hat{m} \;+\; \underbrace{\left(\frac{1}{\|m\|} - \frac{\alpha}{\|m\|^2}\right)}_{\text{perpendicular coefficient}}\delta_\perp$$

Two effects at second order:

1. **Radial pullback.** The $\hat{m}$ component shrinks by $\|\delta_\perp\|^2 / (2\|m\|^2)$. This is the sphere curvature: spreading perpendicular to the pole pulls the mean resultant vector inward. This is exactly the mechanism behind the spherical variance.

2. **Radial-perpendicular coupling.** The perpendicular coefficient acquires a correction $-\alpha/\|m\|^2$ that depends on the radial fluctuation $\alpha = \hat{m}^\top\delta$. Points with positive radial displacement (further from origin) get *less* angular spread; points with negative radial displacement (closer to origin) get *more*. This is the magnifying-glass effect near the origin, now appearing as a perturbative correction.

## Second-order mean

Taking expectations (using $\mathbb{E}[\delta] = 0$, so $\mathbb{E}[\alpha] = 0$, $\mathbb{E}[\delta_\perp] = 0$):

$$\mathbb{E}[z] \approx \hat{m} - \frac{\mathbb{E}[\|\delta_\perp\|^2]}{2\|m\|^2}\,\hat{m} - \frac{\mathbb{E}[\alpha\,\delta_\perp]}{\|m\|^2}$$

The first correction: $\mathbb{E}[\|\delta_\perp\|^2] = \text{tr}(P_{\perp m}\,\Sigma_x\,P_{\perp m}) = \text{tr}(\Sigma_x) - \hat{m}^\top\Sigma_x\,\hat{m}$, which is the same quantity from Proposition 1. So:

$$\mathbb{E}[z] \approx \left(1 - \frac{\text{tr}(\Sigma_x) - \hat{m}^\top\Sigma_x\,\hat{m}}{2\|m\|^2}\right)\hat{m} - \frac{P_{\perp m}\,\Sigma_x\,\hat{m}}{\|m\|^2}$$

The second correction, $\mathbb{E}[\alpha\,\delta_\perp] = P_{\perp m}\,\Sigma_x\,\hat{m}$, is the off-diagonal block of the covariance between the radial and perpendicular components. It vanishes when $\Sigma_x$ commutes with $P_{\perp m}$ (e.g., when the input covariance is isotropic, or when $\hat{m}$ is an eigenvector of $\Sigma_x$).

So the mean resultant length is less than 1, consistent with the spherical variance bound — and now we have the exact correction.

## Second-order covariance

This is where it gets messier. The fluctuation around the corrected mean is:

$$\varepsilon = z - \mathbb{E}[z] = \frac{1}{\|m\|}\delta_\perp - \frac{1}{\|m\|^2}\left[\alpha\,\delta_\perp - \mathbb{E}[\alpha\,\delta_\perp]\right] - \frac{1}{2\|m\|^2}\left[\|\delta_\perp\|^2 - \mathbb{E}[\|\delta_\perp\|^2]\right]\hat{m} + O(\|\delta\|^3/\|m\|^3)$$

The covariance $\Sigma_z = \mathbb{E}[\varepsilon\,\varepsilon^\top]$ has:

- **Leading term** $O(1/\|m\|^2)$: same as before, $\frac{1}{\|m\|^2}P_{\perp m}\,\Sigma_x\,P_{\perp m}$

- **Cross terms** $O(1/\|m\|^3)$: these involve $\mathbb{E}[\delta_\perp \cdot \alpha\,\delta_\perp^\top]$ and its transpose, which are third central moments of $\delta$. **These vanish for any distribution symmetric about $\mu$** (e.g., Gaussian inputs).

- **Quadratic terms** $O(1/\|m\|^4)$: these involve fourth moments like $\mathbb{E}[\alpha^2\,\delta_\perp\,\delta_\perp^\top]$ and $\mathbb{E}[\|\delta_\perp\|^4]$. These are nonzero even for Gaussian inputs and would give the second-order covariance correction, but the expressions involve fourth-moment tensors of $\delta$ that are cumbersome to write in general.

## What the second order buys you

For the mean: a concrete, always-nonzero correction that makes $\|\mathbb{E}[z]\| < 1$, connecting directly to the spherical variance. The radial pullback $\text{tr}(P_\perp \Sigma_x P_\perp)/(2\|m\|^2)$ is the leading contribution to spherical variance within the Taylor framework, confirming consistency with the global geometric bound.

For the covariance: the $O(1/\|m\|^3)$ correction vanishes for symmetric distributions, so the first-order covariance is actually accurate to $O(1/\|m\|^4)$ in that case — better than the naive $O(1/\|m\|^3)$ error estimate suggests. For asymmetric distributions, the correction depends on third moments and could go either way.

