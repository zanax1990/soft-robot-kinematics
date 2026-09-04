# Mathematical model

Let `s ∈ [0, 1]` denote a normalized material coordinate and `L` the physical centerline length. The reference centerline is `X0(s)` and the reconstructed configuration is `x(s, t)`. Time appears as a parameter through a prescribed curvature field `κ(s, t)`; the current implementation evaluates one configuration at a time.

For a planar, unshearable centerline, the tangent angle is

```text
θ(s, t) = θ0 + L ∫₀ˢ κ(ξ, t) dξ.
```

The position follows from tangent integration:

```text
x(s, t) = x0 + L ∫₀ˢ [cos θ(ξ, t), sin θ(ξ, t)]ᵀ dξ.
```

The numerical implementation applies cumulative trapezoidal integration to both equations. It assumes that material points correspond across the reference and deformed centerlines.

## Metrics

For corresponding samples `Xi` and `xi`, pointwise displacement is `di = ||xi - Xi||₂`. The repository reports mean, maximum, and endpoint displacement, together with

```text
shape RMSE = sqrt(mean(di²)).
```

Curvature difference is the root-mean-square difference between sampled curvature fields. Polyline arc length is used only for a numerical consistency check: its relative error is measured against the prescribed physical length `L`. This error reflects discretization of a continuous inextensible centerline; it is not an extensional strain or a general deformation metric.

## Scope

The model is intentionally limited to planar centerline geometry with prescribed curvature. It does not currently solve force balance, constitutive behavior, contact, hysteresis, or three-dimensional rod mechanics.
